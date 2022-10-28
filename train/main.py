from HaloEGNNFlows.train import *
import torch.multiprocessing as mp
import torch

import argparse
from functools import partialmethod
from tqdm import tqdm
import os
import glob

if __name__ == "__main__":
    mp.set_start_method("spawn")
    world_size = torch.cuda.device_count()
    parser = argparse.ArgumentParser()

    parser.add_argument("-d",
                        "--data_location",
                        help='the directory where the data is located',
                        type=str,
                        default="/scratch/y89/kt9438/TNG300_preprocessed_data")

    parser.add_argument("-lr",
                        help="learning rate",
                        type=float,
                        default =1e-3)
    parser.add_argument("-b",
                        "--batch_size",
                        help='Batch size for training',
                        default = 512,
                        type=int)
    parser.add_argument("--max_epochs",
                        type=int,
                        default = 1000)
    parser.add_argument("--ode_reg",
                        type=float,
                        default = 1e-2)
    parser.add_argument("--restart_path",
                        default=None,
                        help='restarting from checkpoint')
    parser.add_argument("--normalize",
                        default='quantile',
                        help='transform the condition columns with sklearn preprocessing transformer')
    parser.add_argument("--random_seed",
                        default=42,
                        type=int,
                        help='random seed')
    parser.add_argument("--patience",
                        default=3,
                        type=int,
                        help='patience for learning rate scheduler')
    parser.add_argument("--log_dir",
                        default="/home/196/kt9438/log_dir",
                        help='checkpoint log path')

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
        "SubhaloMass",
        "SubhaloSFR",
        "SubhaloVmax",
        "SubhaloVelDisp",
    ]

    n_dims = len(position_columns)
    in_node_nf = len(feature_columns)
    context_node_nf = len(condition_columns)

    # replace the parameters with the params pkl value if we are restarting

    if args.restart_path is not None:
        restart_params = np.load(
            os.path.join( os.path.dirname(args.restart_path), "params.pkl"),
            allow_pickle=True
        )
        batch_size = restart_params['batch_size']
        max_epochs = restart_params['max_epochs']
        ode_reg    = restart_params['ode_regularization']
        random_seed = restart_params['random_seed']
        patience = restart_params['patience']
        lr = restart_params['lr']
        restart_path = args.restart_path
    else:
        batch_size = args.batch_size
        max_epochs = args.max_epochs
        ode_reg = args.ode_reg
        random_seed = args.random_seed
        patience = args.patience
        restart_path = None
        lr = args.lr


    filelist_npy = None if not os.path.exists("valid_files.npy") else "valid_files.npy"
    transformer_pkl = (
        None
        if not os.path.exists("condition_normalizer.pkl")
        else "condition_normalizer.pkl"
    )
    #filelist_npy=None
    #transformer_pkl=None

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
        transform_type=args.normalize
    )

    # Multi GPU version is not working just yet!
    # check if there are more than one gpu
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        backend = "nccl"
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
                    ode_reg,
                    # The "scalar" feature bounded to the progenitors
                    feature_columns,
                    # The "positional" feature bounded to the progenitors
                    position_columns,
                    # initialize the list condtional columns at redshift 0
                    condition_columns,
                    full_columns_names,
                    batch_size,
                    lr,
                    restart_path,
                    backend,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train_loop(
            valid_files,
            max_epochs,
            condition_normalizer,
            ode_reg,
            # The "scalar" feature bounded to the progenitors
            feature_columns,
            # The "positional" feature bounded to the progenitors
            position_columns,
            # initialize the list condtional columns at redshift 0
            condition_columns,
            full_columns_names,
            restart_path=restart_path,
            batch_size=batch_size,
            lr=lr,
            random_seed=random_seed,
            patience=patience,
            log_dir=args.log_dir
        )

