import os
import torch
import logging
import argparse
import numpy as np

from run_evaluation import (
    BASE_DIR,
    MODEL_BASE_DIR,
    DATA_BASE_DIR,
    GENERATIONS_BASE_DIR,
    MODELS_TO_USE,
    log_and_print, 
    read_test_set
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, help='File with train sentences/inps to get switch', default='/experiment1_speeds_30000.txt')
    parser.add_argument('--test_file', type=str, help='File with test sentences/inps', default='/eval.txt')
    parser.add_argument('--logging_file', type=str, help='File to log to', default='./evaluation_log')
    parser.add_argument('--limit', type=int, help='Num of samples to use', default=1000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename=os.path.join(GENERATIONS_BASE_DIR, args.logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger('evaluation_logger')

    inps, targets, deltas = read_test_set(args)
    
    with open(GENERATIONS_BASE_DIR + args.pred_file) as f:
        pred_speeds = f.readlines()

    num_valid_samples = 0
    generated_deltas = []
    for i in range(0, len(pred_speeds), 2):
        inp, s1 = pred_speeds[0].strip().split('\t')
        gen, s2 = pred_speeds[1].strip().split('\t')

        if s1 != 0 and s2 != 0:
            num_valid_samples += 1
        generated_deltas.append(float(s2) - float(s1))

    mae = np.sum(np.absolute((np.array(generated_deltas) - np.array(deltas)))) / num_valid_samples
    print(f'Mean Absolute Error: {mae}')

