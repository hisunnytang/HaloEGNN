#!/bin/bash
#PBS -l ncpus=14
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=128GB
#PBS -l storage=scratch/dg97+scratch/y89
#PBS -q gpursaa
#PBS -P dg97
#PBS -l walltime=48:00:00
#PBS -N r38734815_b128rm2expz
#PBS -l jobfs=100GB

source ~/.bashrc
conda init bash
conda activate pytorch

cd $PBS_JOBFS
cp /scratch/y89/kt9438/preprocessed.zip .
unzip -q preprocessed.zip
#/home/196/kt9438/HaloEGNN/train/log/38734815.gadi-pbs/egnn_37_val=17.457.pt
cp $PBS_O_WORKDIR/main.py .
#python main.py -d TNG300_preprocessed_data -lr 1e-3 -b 128 --max_epochs 1000 --ode_reg 1e-2 --normalize power --log_dir $PBS_O_WORKDIR/log  > $PBS_O_WORKDIR/$PBS_JOBNAME.log 
python main.py -d TNG300_preprocessed_data --restart_path "/home/196/kt9438/HaloEGNN/train/log/38734815.gadi-pbs/egnn_37_val=17.457.pt" --log_dir $PBS_O_WORKDIR/log  > $PBS_O_WORKDIR/$PBS_JOBNAME.log 
