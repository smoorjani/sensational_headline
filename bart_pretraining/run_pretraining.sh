#!/bin/bash
#SBATCH --mem=93g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4 # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --gpus=4
#SBATCH --account=bblr-delta-gpu
#SBATCH --job-name=bart_pretraining
#SBATCH --output="bart_pretrain.%j.%N.out"
#SBATCH --no-requeue
#SBATCH --time=48:00:00      # hh:mm:ss for the job

# module reset # drop modules and explicitly load the ones needed
# module load default python modtree/gpu openmpi/4.1.2 ucx/1.11.2 gcc/11.2.0 anaconda3_cpu

# module load anaconda3
# conda init bash
eval "$(conda shell.bash hook)"
conda activate sen2

cd /u/smoorjani/control_tuning/sensational_headline/bart_pretraining

export HF_DATASETS_CACHE=/projects/bblr/smoorjani/huggingface_cache
export MODEL_TYPE="bart-base"
export BOS=1

if [ $BOS == 0 ]; then
    python bart_pretrainer.py \
        --model_name_or_path facebook/$MODEL_TYPE \
        --train_file /projects/bblr/smoorjani/control_tuning/yelpdata/pretraining_data/train.json \
        --validation_file /projects/bblr/smoorjani/control_tuning/yelpdata/pretraining_data/val.json \
        --output_dir /projects/bblr/smoorjani/control_tuning/pretrained_bart/$MODEL_TYPE \
        --overwrite_output_dir \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=8 \
        --do_train --do_eval \
        --predict_with_generate \
        --learning_rate=3e-5 \
        --num_train_epochs=3 \
        --max_source_length=256 \
        --max_target_length=256 \
        --val_max_target_length=256 \
        --save_steps 100000
else
    python bart_pretrainer.py \
        --model_name_or_path facebook/$MODEL_TYPE \
        --train_file /projects/bblr/smoorjani/control_tuning/yelpdata/pretraining_data/train.json \
        --validation_file /projects/bblr/smoorjani/control_tuning/yelpdata/pretraining_data/val.json \
        --output_dir /projects/bblr/smoorjani/control_tuning/pretrained_bart/bos_$MODEL_TYPE \
        --overwrite_output_dir \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=8 \
        --do_train --do_eval \
        --predict_with_generate \
        --learning_rate=3e-5 \
        --num_train_epochs=3 \
        --max_source_length=256 \
        --max_target_length=256 \
        --val_max_target_length=256 \
        --forced_bos_token "<s>" \
        --save_steps 100000
fi




