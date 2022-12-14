#!/bin/bash
#SBATCH --acount=rrg-aspuru
#SBATCH --cpus-per-task=10
#SBATCH --mem=12000M 
#SBATCH --time=0-10:00
#SBATCH --output=gpmol.log
#SBATCH --gres=gpu:1

module load python/3.8
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.03.3

source ~/env/gpmol/bin/activate

DATASET=$1
GOAL=$2
MODEL=$3
FEATURE=$4
ACQ_FUNC=$5
# BETA=$6

time python run.py $DATASET $GOAL $MODEL $FEATURE $ACQ_FUNC # $BETA

deactivate
