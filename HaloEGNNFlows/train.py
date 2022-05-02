from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from torch.utils.data import DataLoader
import torch
import numpy as np
import glob
import os
import pickle
import datetime
import random
import argparse

from .EGNNFlows.models import get_model
from .EGNNFlows.datasets import ProgenitorDataset
from .EGNNFlows.flows.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
)
from .EGNNFlows.train import train_step, val_step
from .EGNNFlows.utils import subtract_the_boundary
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from collections import OrderedDict


def prepare_dataloaders(
    filelist,
    condition_columns,
    data_columns,
    position_columns=["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
    feature_columns=["SubhaloMassType_1", "SubhaloMergeRedshift"],
    max_progenitors=20,
    initial_slice=0,
    final_slice=1,
    batch_size=32,
    num_workers=0,
    random_seed=42,
    train_test_split=[0.8, 0.1, 0.1],
    shuffle_train=False,
    distributed=False,
    world_size=None,
    rank=None,
):

    dataset = ProgenitorDataset(
        filelist,
        condition_columns=condition_columns,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=max_progenitors,
        initial_slice=initial_slice,
        final_slice=final_slice,
        data_columns=data_columns,
    )

    train_size = int(len(dataset) * train_test_split[0])
    val_size = int(len(dataset) * train_test_split[1])
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    if not distributed:
        dl_train = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_train,
        )
        dl_val = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dl_test = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_train,
            drop_last=True,
        )
        dl_train = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            sampler=sampler,
        )
        dl_val = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dl_test = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return dl_train, dl_val, dl_test


def obtain_condition_transformer(dl, max_samples=500, transform_type="power"):
    assert transform_type in ["power", "quantile"]
    count = 0
    targets = []
    for _, input_cond in dl:
        targets.append(input_cond[0])
        if count > max_samples:
            break
        count += 1
    targets = torch.cat(targets)
    if transform_type == "power":
        condition_normalizer = PowerTransformer().fit(targets)
    elif transform_type == "quantile":
        condition_normalizer = QuantileTransformer(output_distribution="normal").fit(
            targets
        )

    return targets, condition_normalizer


def prepare_filelist_and_transformer(
    filelist_npy=None,
    transformer_pkl=None,
    filelist=[],
    condition_columns=[],
    full_columns_names=[],
    max_progenitors=20,
    initial_slice=0,
    final_slice=1,
    batch_size=512,
    num_workers=0,
    transform_type="quantile",
):
    """pre-calculate existing files/ and transformer
    - it would load existing precalculations/ else compute from scratch

    return:
        filelist: list of valid files with number of progenitors >= 2
        condition_transformer: power transformer for the condition
    """
    if filelist_npy is None or transformer_pkl is None:
        failed_files = validate_datasets(
            filelist,
            condition_columns,
            full_columns_names,
            max_progenitors=max_progenitors,
            initial_slice=initial_slice,
            final_slice=final_slice,
            batch_size=batch_size,
            num_workers=0,
        )
        valid_files = sorted(list(set(filelist) - set(failed_files)))
        dl_train, dl_val, dl_test = prepare_dataloaders(
            valid_files,
            condition_columns,
            full_columns_names,
            max_progenitors=max_progenitors,
            initial_slice=initial_slice,
            final_slice=final_slice,
            batch_size=batch_size,
            random_seed=42,
            train_test_split=[0.8, 0.1, 0.1],
            shuffle_train=False,
        )
        sample_condition, condition_normalizer = obtain_condition_transformer(
            dl_train, max_samples=10000, transform_type=transform_type
        )
        np.save("valid_files.npy", valid_files)
        pickle.dump(condition_normalizer, open(f"condition_normalizer.pkl", "wb"))
    else:
        valid_files = np.load(filelist_npy)
        condition_normalizer = pickle.load(open(transformer_pkl, "rb"))

    return valid_files, condition_normalizer


def checkpoint_model(
    model,
    scheduler,
    optimizer,
    train_history,
    val_history,
    epoch=0,
    log_path=None,
):
    if isinstance(model, DDP):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_history,
                "val_loss": val_history,
            },
            f"{log_path}/egnn_{epoch}_val={val_history[epoch]:.3f}.pt",
        )
    else:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_history,
                "val_loss": val_history,
            },
            f"{log_path}/egnn_{epoch}_val={val_history[epoch]:.3f}.pt",
        )
    print(f"Checkpoint: {log_path}/egnn_{epoch}_val={val_history[epoch]:.3f}.pt")


def validate_datasets(
    filelist,
    condition_columns,
    data_columns,
    position_columns=["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
    feature_columns=["SubhaloMassType_1", "SubhaloMergeRedshift"],
    max_progenitors=20,
    initial_slice=0,
    final_slice=1,
    batch_size=32,
    num_workers=4,
):
    device = "cpu"

    dataset = ProgenitorDataset(
        filelist,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=max_progenitors,
        initial_slice=initial_slice,
        final_slice=final_slice,
        condition_columns=condition_columns,
        data_columns=data_columns,
    )

    # overriding the original dataset
    class IdxProgDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return idx, self.dataset[idx]

    idx_dataset = IdxProgDataset(dataset)
    dl_full = DataLoader(
        idx_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    # validate the full data, and assert there is NO invalid data with normalization
    count = 0
    failed_files = []
    for idxes, (input_graph, input_cond) in dl_full:
        x, h, node_mask, edge_mask = input_graph
        context = input_cond[0]

        x = x.to(device)
        h = h.to(device)
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)
        context = context.to(device)

        subtract_the_boundary(x, node_mask)
        xx = remove_mean_with_mask(x, node_mask)

        # h = log_normalize_with_mask(h, node_mask)

        h = (h + 1e-8).log() * node_mask
        # dont normalize here
        # context = log_norm_target(context)

        x_norm = 300.0
        # center the position coordinate
        xx = remove_mean_with_mask(x / x_norm, node_mask)

        if torch.isnan(xx).sum() > 0:
            idx_in_batch = torch.where(torch.isnan(xx))[0].unique()
            # print(idxes[idx_in_batch])
            # print(filelist[idxes[idx_in_batch]])
            # assert_mean_zero_with_mask(xx, node_mask)
            node_mask.sum(-1)
            # print(ncounts[idx_in_batch])
            for i in idx_in_batch:
                failed_files.append(filelist[idxes[i]])
        count += 1
    return failed_files


def transform_z_to_scale(z_idx):
    def transform_graph(input_graph):
        # transform only the features
        if z_idx is None:
            return input_graph
        x, h, node_mask, edge_mask = input_graph
        h[:, :, z_idx] = 1 / (1 + h[:, :, z_idx])
        h[:, :, z_idx] = (h[:, :, z_idx]).exp()
        return x, h, node_mask, edge_mask

    return transform_graph


def create_log_directory(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    jobid = os.environ["PBS_JOBID"]
    if jobid is None:
        # use current time as dir name
        now = datetime.now()
        jobid = now.strftime("%m%d%Y_%H%M")
    if not os.path.exists(f"{log_dir}/{jobid}"):
        os.makedirs(f"{log_dir}/{jobid}")
    return f"{log_dir}/{jobid}"


def filter_filelist(filelist, ncolumns, min_counts=2):
    gt1 = []
    for f in filelist:
        non_zeros = (np.load(f)[0] == 0).sum() // ncolumns
        if non_zeros >= min_counts:
            gt1.append(f)
    return gt1


# https://datascience.stackexchange.com/questions/66345/why-ml-model-produces-different-results-despite-random-state-defined-and-how-to
def seed_everything(seed=42):
    """ "
    Seed everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_loop(
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
    full_columns_names=[],
    restart_path=None,
    batch_size=256,
    lr=1e-3,
    random_seed=42,
    patience=3,
    log_dir="log_dir",
):

    # initialize random seed across all libs
    seed_everything(random_seed)

    n_dims = len(position_columns)
    in_node_nf = len(feature_columns)
    context_node_nf = len(condition_columns)
    params = locals()

    # Prepare dataloaders
    dl_train, dl_val, dl_test = prepare_dataloaders(
        valid_files,
        condition_columns,
        full_columns_names,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=batch_size,
        num_workers=0,
        random_seed=random_seed,
        train_test_split=[0.8, 0.1, 0.1],
        shuffle_train=True,
        distributed=False,
    )
    # Prepare Models and Priors/ Optimizer/ LR scheduler
    prior, flow = get_model(
        in_node_nf=in_node_nf,  # Number of Features to fit (i.e. Progenitor Halo Mass)
        dynamics_in_node_nf=1,  # Use Time as additional Feature
        context_node_nf=context_node_nf,  # Number of Conditional Features
        n_dims=n_dims,  # Number of "Equivariant" Dimension
    )
    device = "cuda:0"
    flow = flow.to(device)
    #optim = torch.optim.Adam(
    #    flow.parameters(),
    #    lr=lr,
    #)
    optim = torch.optim.AdamW(
        flow.parameters(),
        lr=lr,
#        amsgrad=True,
#        weight_decay=1e-12
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, verbose=1, mode="min", min_lr=1e-8, patience=patience
    )

    # Load from the restart path if needed
    if restart_path is not None:
        map_location = {"cuda:0": f"cuda:0"}
        ckpt_loc = restart_path
        ckpt = torch.load(ckpt_loc, map_location=map_location)
        print(device, map_location, ckpt_loc)
        flow.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        current_epoch = ckpt["epoch"]
        train_loss = ckpt["loss"]
        val_loss = ckpt["val_loss"]
        max_epochs = len(train_loss)
    else:
        train_loss = np.zeros(max_epochs)
        val_loss = np.zeros(max_epochs)
        current_epoch = 0

    # Apply pre-feature transformation
    zidx = None
    if "SubhaloMergeRedshift" in feature_columns:
        zidx = feature_columns.index("SubhaloMergeRedshift")

    # Initialize log path
    if restart_path is None:
        log_path = create_log_directory(log_dir)
        pickle.dump(condition_normalizer, open(f"{log_path}/scaler.pkl", "wb"))
    else:
        log_path = os.path.dirname(restart_path)

    # Save the hyperparameters as pickle
    params["log_path"] = log_path
    if restart_path is None:
        with open(os.path.join(log_path, "params.pkl"), "wb") as p:
            pickle.dump(params, p)

    # Training Loop
    for i in range(current_epoch, max_epochs):
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

        val = val_step(
            flow,
            prior,
            dl_val,
            condition_normalizer,
            device=device,
            ode_regularization=ode_regularization,
            transform_input=transform_z_to_scale(zidx),
        )

        train_loss[i] = loss.item()
        val_loss[i] = val.item()
        checkpoint_model(
            flow, scheduler, optim, train_loss, val_loss, epoch=i, log_path=log_path
        )
        print(f"Epoch {i}: train loss = {loss.item():.2f}; val loss = {val.item():.2f}")
        scheduler.step(val)


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
    full_columns_names=[],
    restart_path=None,
    batch_size=256,
    lr=1e-3,
    random_seed=42,
    patience=3,
):

    # initialize random seed across all libs
    seed_everything(random_seed)

    n_dims = len(position_columns)
    in_node_nf = len(feature_columns)
    context_node_nf = len(condition_columns)
    params = locals()

    # Prepare dataloaders
    dl_train, dl_val, dl_test = prepare_dataloaders(
        valid_files,
        condition_columns,
        full_columns_names,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=batch_size,
        num_workers=0,
        random_seed=random_seed,
        train_test_split=[0.8, 0.1, 0.1],
        shuffle_train=False,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
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
    #optim = torch.optim.Adam(
    #    flow.parameters(),
    #    lr=lr,
    #)
    optim = torch.optim.AdamW(
        flow.parameters(),
        lr=lr,
#        amsgrad=True,
#        weight_decay=1e-12
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, verbose=1, mode="min", min_lr=1e-8, patience=patience
    )

    # initialize model on rank0
    #    actnorm = flow.module.transformations[0]
    #    input_graph, input_cond =next(iter(dl_train))
    #    if rank == 0:
    #        flow_forward(flow, prior, input_graph, input_cond, device=device,)

    # copy the parameters in actonorm on rank0 to the rest replicas
    #    dist.broadcast(actnorm.h_t.data, src=0, group=dist.new_group(list(range(world_size))))
    #    dist.broadcast(actnorm.x_log_s.data, src=0, group=dist.new_group(list(range(world_size))))
    #    dist.broadcast(actnorm.h_log_s.data, src=0, group=dist.new_group(list(range(world_size))))
    #    dist.broadcast(actnorm.initialized.data, src=0, group=dist.new_group(list(range(world_size))))
    # assert that all other rank shows the same copy
    #    print(actnorm.initialized.data, device)
    #    print(actnorm.h_t.data, device)
    #    print(actnorm.x_log_s.data, device)
    #    print(actnorm.h_log_s.data, device)

    #    input_graph, input_cond =next(iter(dl_train))
    #    out = flow_forward(flow, prior, input_graph, input_cond, device=device,)
    #    print(out, device)
    if restart_path is not None:
        map_location = {"cuda:0": f"cuda:{rank}"}
        ckpt_loc = restart_path
        ckpt = torch.load(ckpt_loc, map_location=map_location)
        print(rank, map_location, ckpt_loc)
        # adding prefix to all the parameters in a state_dict
        state_dict = OrderedDict()
        for k, v in ckpt["model_state_dict"].items():
            state_dict[f"module.{k}"] = v
        print(flow.load_state_dict(state_dict, strict=True))
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        current_epoch = ckpt["epoch"]
        train_loss = ckpt["loss"]
        val_loss = ckpt["val_loss"]
        max_epochs = len(train_loss)
    else:
        train_loss = np.zeros(max_epochs)
        val_loss = np.zeros(max_epochs)
        current_epoch = 0

    zidx = None
    # if "SubhaloMergeRedshift" in feature_columns:
    #    zidx = feature_columns.index("SubhaloMergeRedshift")

    if rank == 0:
        if restart_path is None:
            log_path = create_log_directory("log_dir")
            pickle.dump(condition_normalizer, open(f"{log_path}/scaler.pkl", "wb"))
        else:
            log_path = os.path.dirname(restart_path)
        # Save the hyperparameters as pickle
        params["log_path"] = log_path
        with open(os.path.join(log_path, "params.pkl"), "wb") as p:
            pickle.dump(params, p)

    for i in range(current_epoch, max_epochs):
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
            tqdm.write(
                f"Epoch {i}: train loss = {loss.item()/ world_size:.2f}; val loss = {val.item():.2f}"
            )
        else:
            val = torch.zeros(1, device=device)
        dist.broadcast(val, src=0, group=dist.new_group(list(range(world_size))))
        scheduler.step(val)

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
    full_columns_names=[],
    batch_size=256,
    lr=1e-3,
    restart_path=None,
    backend="nccl",
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "38500"
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
        full_columns_names,
        restart_path,
        batch_size,
        lr,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_location",
        help="the directory where the data is located",
        type=str,
        default="/scratch/y89/kt9438/TNG300_preprocessed_data",
    )

    parser.add_argument("-lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for training", default=256, type=int
    )
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--ode_reg", type=float, default=1e-2)
    parser.add_argument(
        "--restart_path", default=None, help="restarting from checkpoint"
    )
    parser.add_argument(
        "--normalize",
        default="power",
        help="transform the condition columns with sklearn preprocessing transformer",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="patience for the ReduceLROnPleatau"
    )

    args = parser.parse_args()
    # location when the TNG300 preprocessed data is
    preprocessed_loc = args.data_location
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
        "DMFrac",
        "GasFrac",
        "StarWindFrac",
        "BHFrac",
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
        num_workers=56,
        transform_type=args.normalize,
    )
    print(len(valid_files))

    train_loop(
        valid_files,
        args.max_epochs,
        condition_normalizer,
        args.ode_reg,
        # The "scalar" feature bounded to the progenitors
        feature_columns,
        # The "positional" feature bounded to the progenitors
        position_columns,
        # initialize the list condtional columns at redshift 0
        condition_columns,
        full_columns_names,
        restart_path=args.restart_path,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
