#!/bin/bash
#PBS -l ncpus=14
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=128GB
#PBS -l storage=scratch/dg97+scratch/y89
#PBS -q gpursaa
#PBS -P dg97
#PBS -l walltime=48:00:00
#PBS -N mcmc
#PBS -l jobfs=100GB

source ~/.bashrc
conda init bash
conda activate pytorch

cd $PBS_JOBFS
cp /scratch/y89/kt9438/preprocessed.zip .
unzip -q preprocessed.zip

cp $PBS_O_WORKDIR/infer_graph.py .
python infer_graph.py &> $PBS_O_WORKDIR/$PBS_JOBNAME.log 
