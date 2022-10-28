#!/bin/bash
#PBS -l ncpus=14
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=128GB
#PBS -l storage=scratch/dg97+scratch/y89
#PBS -q gpursaa
#PBS -P dg97
#PBS -l walltime=48:00:00

#PBS -N job_EGNNFlows


source ~/.bashrc
cd $PBS_O_WORKDIR
conda init bash
conda activate pytorch
python train.py > $PBS_JOBID.log 

