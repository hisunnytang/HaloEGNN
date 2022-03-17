from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader
import torch
import numpy as np
import glob
import os
import pickle

from HaloEGNNFlows.EGNNFlows.models import get_model
from HaloEGNNFlows.EGNNFlows.flow_forward import train_step, val_step
from HaloEGNNFlows.EGNNFlows.datasets import ProgenitorDataset
from HaloEGNNFlows.EGNNFlows.flow_forward import flow_forward

from HaloEGNNFlows.EGNNFlows.flows.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
)
from HaloEGNNFlows.EGNNFlows.utils import subtract_the_boundary
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist


from train import (
    prepare_dataloaders,
    obtain_condition_transformer,
    checkpoint_model,
    validate_datasets,
    transform_z_to_scale,
    create_log_directory,
    prepare_filelist_and_transformer,
)


def dataloaders_parallel(rank, world_size, valid_files):
    # location when the TNG300 preprocessed data is
    preprocessed_loc = "/scratch/y89/kt9438/TNG300_preprocessed_data"
    # Prepare the columns names
    full_columns_names = np.load(f"{preprocessed_loc}/subhalo_columns.npy")

    # The "scalar" feature bounded to the progenitors
    feature_columns = ["SubhaloMass", "SubhaloMergeRedshift"]
    # The "positional" feature bounded to the progenitors
    position_columns = ["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"]

    # initialize the list condtional columns at redshift 0
    condition_columns = [
        "SubhaloBHMass",
        "SubhaloBHMdot",
        "SubhaloGasMetallicity",
        "SubhaloStarMetallicity",
        "SubhaloMass",
        "SubhaloSFR",
        "SubhaloVmax",
        "SubhaloVelDisp",
    ]

    # Prepare dataloaders
    dl_train, dl_val, dl_test = prepare_dataloaders(
        valid_files,
        condition_columns,
        full_columns_names,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=32,
        num_workers=0,
        random_seed=42,
        train_test_split=[0.8, 0.1, 0.1],
        shuffle_train=False,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    return dl_train, dl_val, dl_test


# https://github.com/olehb/pytorch_ddp_tutorial/blob/main/ddp_tutorial_multi_gpu.py
def train_parallel(
    rank,
    world_size,
    valid_files,
    max_epochs,
    condition_normalizer,
    ode_regularization=0.01,
    # The "scalar" feature bounded to the progenitors
    feature_columns=["SubhaloMass", "SubhaloMergeRedshift"],
    # The "positional" feature bounded to the progenitors
    position_columns=["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
    # initialize the list condtional columns at redshift 0
    condition_columns=[
        "SubhaloBHMass",
        "SubhaloBHMdot",
        "SubhaloGasMetallicity",
        "SubhaloStarMetallicity",
        "SubhaloMass",
        "SubhaloSFR",
        "SubhaloVmax",
        "SubhaloVelDisp",
    ],
    restart_path=None,
):

    n_dims = len(position_columns)
    in_node_nf = len(feature_columns)
    context_node_nf = len(condition_columns)
    dl_train, dl_val, dl_test = dataloaders_parallel(rank, world_size, valid_files)

    # Prepare Models and Priors/ Optimizer/ LR scheduler
    prior, flow = get_model(
        in_node_nf=in_node_nf,  # Number of Features to fit (i.e. Progenitor Halo Mass)
        dynamics_in_node_nf=1,  # Use Time as additional Feature
        context_node_nf=context_node_nf,  # Number of Conditional Features
        n_dims=n_dims,  # Number of "Equivariant" Dimension
    )
    device = torch.device(f"cuda:{rank}")
    flow = flow.to(device)

    flow = DDP(
        flow,
        device_ids=[rank],
        # output_device=rank,
    )
    optim = torch.optim.AdamW(flow.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, verbose=1, mode="min", min_lr=1e-8
    )
    # restart_path = "log_dir/36795639.gadi-pbs/egnn_0_val=79.959.pt"

    if restart_path is not None:
        map_location = {"cuda:0": f"cuda:{rank}"}
        ckpt_loc = restart_path
        print(rank, map_location, ckpt_loc)
        flow.load_state_dict(
            torch.load(ckpt_loc, map_location=map_location)["model_state_dict"]
        )
        optim.load_state_dict(
            torch.load(ckpt_loc, map_location=map_location)["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            torch.load(ckpt_loc, map_location=map_location)["scheduler_state_dict"]
        )

    zidx = None
    if "SubhaloMergeRedshift" in feature_columns:
        zidx = feature_columns.index("SubhaloMergeRedshift")

    train_loss = np.zeros(max_epochs)
    val_loss = np.zeros(max_epochs)

    if rank == 0:
        if restart_path is None:
            log_path = create_log_directory("log_dir")
        else:
            log_path = os.path.dirname(restart_path)
        pickle.dump(condition_normalizer, open(f"{log_path}/scaler.pkl", "wb"))

    for i in range(max_epochs):
        dl_train.sampler.set_epoch(i)
        loss = train_step(
            flow,
            prior,
            optim,
            dl_train,
            condition_normalizer,
            device=device,
            ode_regularization=ode_regularization,
            transform_input=transform_z_to_scale(zidx),
        )
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        if rank == 0:
            val = val_step(
                flow,
                prior,
                dl_val,
                condition_normalizer,
                device=device,
                ode_regularization=ode_regularization,
                transform_input=transform_z_to_scale(zidx),
            )

            train_loss[i] = loss.item() / world_size
            val_loss[i] = val.item()
            checkpoint_model(
                flow, scheduler, optim, train_loss, val_loss, epoch=i, log_path=log_path
            )
            print(
                f"Epoch {i}: train loss = {loss.item()/ world_size:.2f}; val loss = {val.item():.2f}"
            )
        else:
            val = torch.zeros(1, device=device)
        dist.broadcast(val, src=0, group=dist.new_group(list(range(world_size))))
        scheduler.step(val)

        continue
        # TODO fixed this load state dict part?
        # Make sure all copies over GPU begins with the same params
        # configure map_location properly
        map_location = {"cuda:0": f"cuda:{rank}"}
        ckpt_loc = f"{log_path}/egnn_{i}_val={val.item():.3f}.pt"
        print(rank, map_location, ckpt_loc)
        flow.load_state_dict(
            torch.load(ckpt_loc, map_location=map_location)["model_state_dict"]
        )
        optim.load_state_dict(
            torch.load(ckpt_loc, map_location=map_location)["optimizer_state_dict"]
        )
        print(rank, map_location, ckpt_loc)
        dist.barrier()
        print("after bariier?????")

    cleanup()


def cleanup():
    dist.destroy_process_group()


def init_process(
    rank,
    size,
    fn,
    files,
    max_epochs,
    condition_normalizer,
    ode_regularization=0.01,
    # The "scalar" feature bounded to the progenitors
    feature_columns=["SubhaloMass", "SubhaloMergeRedshift"],
    # The "positional" feature bounded to the progenitors
    position_columns=["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
    # initialize the list condtional columns at redshift 0
    condition_columns=[
        "SubhaloBHMass",
        "SubhaloBHMdot",
        "SubhaloGasMetallicity",
        "SubhaloStarMetallicity",
        "SubhaloMass",
        "SubhaloSFR",
        "SubhaloVmax",
        "SubhaloVelDisp",
    ],
    backend="nccl",
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    fn(
        rank,
        size,
        files,
        max_epochs,
        condition_normalizer,
        ode_regularization,
        feature_columns,
        position_columns,
        condition_columns,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    world_size = torch.cuda.device_count()
    # location when the TNG300 preprocessed data is
    preprocessed_loc = "/scratch/y89/kt9438/TNG300_preprocessed_data"
    # Prepare file list
    filelist = sorted(glob.glob(f"{preprocessed_loc}/prog_sublink_*.npy"))
    # Prepare the columns names
    full_columns_names = np.load(f"{preprocessed_loc}/subhalo_columns.npy")

    # The "scalar" feature bounded to the progenitors
    feature_columns = ["SubhaloMass", "SubhaloMergeRedshift"]
    # The "positional" feature bounded to the progenitors
    position_columns = ["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"]

    # initialize the list condtional columns at redshift 0
    condition_columns = [
        "SubhaloBHMass",
        "SubhaloBHMdot",
        "SubhaloGasMetallicity",
        "SubhaloStarMetallicity",
        "SubhaloMass",
        "SubhaloSFR",
        "SubhaloVmax",
        "SubhaloVelDisp",
    ]

    n_dims = len(position_columns)
    in_node_nf = len(feature_columns)
    context_node_nf = len(condition_columns)

    filelist_npy = None if not os.path.exists("valid_files.npy") else "valid_files.npy"
    transformer_pkl = (
        None
        if not os.path.exists("condition_normalizer.pkl")
        else "condition_normalizer.pkl"
    )

    valid_files, condition_normalizer = prepare_filelist_and_transformer(
        filelist_npy=filelist_npy,
        transformer_pkl=transformer_pkl,
        filelist=filelist,
        condition_columns=condition_columns,
        full_columns_names=full_columns_names,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=512,
        num_workers=0,
    )

    max_epochs = 1000
    ode_regularization = 0.01

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=init_process,
            args=(
                rank,
                world_size,
                train_parallel,
                valid_files,
                max_epochs,
                condition_normalizer,
                ode_regularization,
                # The "scalar" feature bounded to the progenitors
                feature_columns,
                # The "positional" feature bounded to the progenitors
                position_columns,
                # initialize the list condtional columns at redshift 0
                condition_columns,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
