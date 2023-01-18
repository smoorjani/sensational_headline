#!/bin/bash
#SBATCH --mem=93g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4 # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --gpus=2
#SBATCH --account=bblr-delta-gpu
#SBATCH --job-name=sensational_headline
#SBATCH --output="%j.%N.out"
#SBATCH --no-requeue
#SBATCH --time=10:00:00      # hh:mm:ss for the job

# module reset # drop modules and explicitly load the ones needed
# module load default python modtree/gpu openmpi/4.1.2 ucx/1.11.2 gcc/11.2.0 anaconda3_cpu

# module load anaconda3
# conda init bash
eval "$(conda shell.bash hook)"
conda activate sen

cd /u/smoorjani/control_tuning/sensational_headline
export PROJECT=/projects/bblr/smoorjani && export PYTHONPATH="${PYTHONPATH}:/u/smoorjani/control_tuning/speed_control"

deepspeed --num_gpus=2 sensation_generation_trainer.py 