#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/smoorjani/anaconda3/lib
CUDA_LAUNCH_BLOCKING=1 python sensation_generation.py -path sensation_save/Rl/chkpt1 -thd 0.1 -use_rl True -use_s_score 1 -eval_step 500
# python sensation_generation.py -thd 0.1 -use_s_score 1 -eval_step 500
# deepspeed --num_gpus=2 sensation_generation_trainer.py --batch_size 2