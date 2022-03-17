from HaloEGNNFlows.EGNNFlows.models import get_model
from HaloEGNNFlows.EGNNFlows.flow_forward import train_step, val_step, flow_forward
from sklearn.preprocessing import PowerTransformer
from HaloEGNNFlows.EGNNFlows.datasets import ProgenitorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import glob
import os
import pickle

from HaloEGNNFlows.EGNNFlows.flows.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
)
from HaloEGNNFlows.EGNNFlows.utils import subtract_the_boundary
from torch.utils.data.distributed import DistributedSampler


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
    num_workers=4,
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
            train_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
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


def obtain_condition_transformer(dl, max_samples=500):
    count = 0
    targets = []
    for _, input_cond in dl:
        targets.append(input_cond[0])
        if count > max_samples:
            break
        count += 1
    targets = torch.cat(targets)
    condition_normalizer = PowerTransformer().fit(targets)
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
            dl_train, max_samples=10000
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
        return x, h, node_mask, edge_mask

    return transform_graph


def create_log_directory(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if (jobid := os.environ["PBS_JOBID"]) is None:
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


def resume_training(ckpt_path):
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

    # print(len(filelist))
    # filelist = filter_filelist(filelist, len(full_columns_names), min_prog)
    print(len(filelist))

    failed_files = validate_datasets(
        filelist,
        condition_columns,
        full_columns_names,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=256,
        num_workers=14,
    )

    valid_files = sorted(list(set(filelist) - set(failed_files)))
    print(len(valid_files))

    # Prepare dataloaders
    dl_train, dl_val, dl_test = prepare_dataloaders(
        valid_files,
        condition_columns,
        full_columns_names,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=128,
        num_workers=14,
        random_seed=42,
        train_test_split=[0.8, 0.1, 0.1],
        shuffle_train=False,
    )

    zidx = None
    if "SubhaloMergeRedshift" in feature_columns:
        zidx = feature_columns.index("SubhaloMergeRedshift")
        print(zidx, "feature with this index would be transform to 1/(1+z)")

    # Load the checkpoint path
    ckpt = torch.load(ckpt_path)

    in_node_nf = 2
    # TODO: hyperparameters should also be stored in the ckpt
    # INITIALIZE the model
    # Prepare Models and Priors/ Optimizer/ LR scheduler
    prior, flow = get_model(
        in_node_nf=in_node_nf,  # Number of Features to fit (i.e. Progenitor Halo Mass)
        dynamics_in_node_nf=1,  # Use Time as additional Feature
        context_node_nf=context_node_nf,  # Number of Conditional Features
        n_dims=n_dims,  # Number of "Equivariant" Dimension
    )
    optim = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, verbose=1, mode="min", min_lr=1e-8
    )
    ode_regularization = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load state dict from last epoch
    current_epoch = ckpt["epoch"]
    flow.load_state_dict(ckpt["model_state_dict"])
    optim.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    # train_history
    train_loss = ckpt["loss"]
    val_loss = ckpt["val_loss"]
    max_epochs = len(train_loss)

    dirname = os.path.dirname(ckpt_path)
    condition_normalizer = pickle.load(open(os.path.join(dirname, "scaler.pkl"), "rb"))
    log_path = create_log_directory("log_dir")

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

        print(f"Epoch {i}: train loss = {loss:.2f}; val loss = {val:.2f}")
        train_loss[i] = loss
        val_loss[i] = val
        scheduler.step(val)
        checkpoint_model(
            flow, scheduler, optim, train_loss, val_loss, epoch=i, log_path=log_path
        )


if __name__ == "__main__":
    # ckpt_path = "log_dir/36425451.gadi-pbs/egnn_20_val=56.227.pt"
    ckpt_path = "log_dir/36587008.gadi-pbs/egnn_32_val=49.370.pt"
    resume_training(ckpt_path)

"""

if __name__ == "__main__":

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
    min_prog = 2

    #print(len(filelist))
    #filelist = filter_filelist(filelist, len(full_columns_names), min_prog)
    print(len(filelist))

    failed_files = validate_datasets(
        filelist,
        condition_columns,
        full_columns_names,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=256,
        num_workers=14,
    )

    valid_files = sorted(list( set(filelist) -set(failed_files) ))
    print(len(valid_files))

    # Prepare dataloaders
    dl_train, dl_val, dl_test = prepare_dataloaders(
        valid_files,
        condition_columns,
        full_columns_names,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=128,
        num_workers=14,
        random_seed=42,
        train_test_split=[0.8, 0.1, 0.1],
        shuffle_train=False
    )

    input_graph, input_cond = next(iter(dl_train))
    print([input_graph[i].shape for i in range(4)])
    print(input_cond[0].shape)

    # Prepare Models and Priors/ Optimizer/ LR scheduler
    prior, flow = get_model(
        in_node_nf=in_node_nf,  # Number of Features to fit (i.e. Progenitor Halo Mass)
        dynamics_in_node_nf=1,  # Use Time as additional Feature
        context_node_nf=context_node_nf,  # Number of Conditional Features
        n_dims=n_dims,  # Number of "Equivariant" Dimension
    )
    #nll, loss, _, _ = flow_forward(flow, prior, input_graph, input_cond, device='cuda',)
    #flow = torch.nn.parallel.DistributedDataParallel(flow)

    optim = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, verbose=1, mode="min", min_lr=1e-8
    )
    print('model is ready')

    max_epochs = 100
    ode_regularization = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('torch said cuda is available')

    # normalize the input condition
    sample_condition, condition_normalizer = obtain_condition_transformer(
        dl_train,
        max_samples=10000
    )

    zidx = None
    if "SubhaloMergeRedshift" in feature_columns:
        zidx = feature_columns.index("SubhaloMergeRedshift")
        print(zidx, "feature with this index would be transform to 1/(1+z)")

    train_loss = np.zeros(max_epochs)
    val_loss = np.zeros(max_epochs)

    log_path = create_log_directory('log_dir')
    # dump seralized condtion_normalizer

    pickle.dump(condition_normalizer, open(f"{log_path}/scaler.pkl", 'wb'))

    for i in range(max_epochs):
        loss = train_step(
            flow,
            prior,
            optim,
            dl_train,
            condition_normalizer,
            device=device,
            ode_regularization=ode_regularization,
            transform_input= transform_z_to_scale(zidx)
        )
        val = val_step(
            flow,
            prior,
            dl_val,
            condition_normalizer,
            device=device,
            ode_regularization=ode_regularization,
            transform_input= transform_z_to_scale(zidx)
        )

        train_loss[i] = loss
        val_loss[i] = val
        scheduler.step(val)
        checkpoint_model(
            flow,
            scheduler,
            optim,
            train_loss,
            val_loss,
            epoch=i,
            log_path=log_path
        )
        print(f"Epoch {i}: train loss = {loss:.2f}; val loss = {val:.2f}")
"""
