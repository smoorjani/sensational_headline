#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/smoorjani/anaconda3/lib
python sensation_generation.py -path save/PointerAttn/Pointer_Gen/ -sensation_scorer_path save/sensation/512_0.9579935073852539/ -thd 0.1 -use_rl True -use_s_score 1

#CUDA_LAUNCH_BLOCKING=1 python sensation_generation.py -path save/PointerAttn/Pointer_Gen/ -sensation_scorer_path save/sensation/512_0.9579935073852539/ -thd 0.1 -use_rl True -use_s_score 1
