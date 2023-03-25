#!/bin/bash
#SBATCH --mem=93g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4 # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --gpus=4
#SBATCH --account=bblr-delta-gpu
#SBATCH --job-name=sensational_headline
#SBATCH --output="train.%j.%N.out"
#SBATCH --no-requeue
#SBATCH --time=24:00:00      # hh:mm:ss for the job

# module reset # drop modules and explicitly load the ones needed
# module load default python modtree/gpu openmpi/4.1.2 ucx/1.11.2 gcc/11.2.0 anaconda3_cpu

# module load anaconda3
# conda init bash
eval "$(conda shell.bash hook)"
conda activate sen2

cd /u/smoorjani/control_tuning/sensational_headline
export PYTHONPATH="${PYTHONPATH}:/u/smoorjani/control_tuning/speed_control:/u/smoorjani/control_tuning"

export MASTER_ADDR=$[ $RANDOM % 30000 + 29500 ]

deepspeed --num_gpus=4 --master_port $MASTER_ADDR sensation_generation_trainer.py --total_steps 10000 --lr 5e-5 --gamma 0.1 --batch_size 32 --use_discriminator --control_method tag --discriminator_path /projects/bblr/smoorjani/control_tuning/speed_control/bart/checkpoint_6_0_False_5e-05_2_0.8_3_0.001 --notes "decoder_input_ids"