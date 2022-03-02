from HaloEGNN.EGNNFlows.models import get_model
from HaloEGNN.EGNNFlows.flow_forward import train_step, val_step
from sklearn.preprocessing import PowerTransformer
from HaloEGNN.EGNNFlows.datasets import ProgenitorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np


def prepare_dataloaders(
    filelist,
    condition_columns,
    position_columns=["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"],
    feature_columns=["SubhaloMassType_1", "SubhaloMergeRedshift"],
    max_progenitors=20,
    initial_slice=0,
    final_slice=1,
    batch_size=32,
    num_workers=4,
    random_seed=42,
    train_test_split=[0.8, 0.1, 0.1],
):

    dataset = ProgenitorDataset(
        filelist,
        position_columns=position_columns,
        feature_columns=feature_columns,
        max_progenitors=max_progenitors,
        initial_slice=initial_slice,
        final_slice=final_slice,
        conditon_columns=condition_columns,
    )
    train_size = int(len(dataset) * train_test_split[0])
    val_size = int(len(dataset) * train_test_split[1])
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    dl_train = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_val = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    dl_test = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    return dl_train, dl_val, dl_test


def obtain_condition_transformer(dl, max_sample=500):
    count = 0
    targets = []
    for _, input_cond in dl:
        targets.append(input_cond[0])
        if count > max_sample:
            break
        count += 1
    targets = torch.cat(targets)
    condition_normalizer = PowerTransformer().fit(targets)
    return targets, condition_normalizer


def checkpoint_model(
    model,
    scheduler,
    optimizer,
    train_history,
    val_history,
    epoch=0,
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
        f"egnn_{epoch}_val={val_history[-1]:.3f}.pt",
    )


if __name__ == "__main__":

    # initialize the list of data to model
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

    feature_columns = ["SubhaloMass", "SubhaloMergeRedshift"]

    position_columns = ["SubhaloPos_0", "SubhaloPos_1", "SubhaloPos_2"]

    in_node_nf = len(feature_columns)
    n_dims = len(position_columns)
    context_node_nf = len(condition_columns)

    # Prepare dataloaders
    dl_train, dl_val, dl_test = prepare_dataloaders(
        filelist,
        condition_columns,
        max_progenitors=20,
        initial_slice=0,
        final_slice=1,
        batch_size=32,
        num_workers=4,
        random_seed=42,
        train_test_split=[0.8, 0.1, 0.1],
    )

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

    max_epochs = 100
    ode_regularization = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # normalize the input condition
    sample_condition, condition_normalizer = obtain_condition_transformer(
        dl_train,
        max_samples=10000
    )

    train_loss = np.zeros(max_epochs)
    val_loss = np.zeros(max_epochs)
    for i in range(max_epochs):
        loss = train_step(
            flow,
            prior,
            optim,
            dl_train,
            condition_normalizer,
            device=device,
            ode_regularization=ode_regularization,
        )
        val = val_step(
            flow,
            prior,
            dl_val,
            condition_normalizer,
            device=device,
            ode_regularization=ode_regularization,
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
        )
        print(f"Epoch {i}: train loss = {loss:.2f}; val loss = {val:.2f}")
