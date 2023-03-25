import os
# import torch
import logging
import argparse
import numpy as np
import pandas as pd

from collections import defaultdict

from dutils.function import get_default_switch

# from run_evaluation import (
#     BASE_DIR,
#     MODEL_BASE_DIR,
#     DATA_BASE_DIR,
#     GENERATIONS_BASE_DIR,
#     MODELS_TO_USE,
#     log_and_print, 
#     read_test_set
# )


BASE_DIR = '/projects/bblr/smoorjani/control_tuning'

MODEL_BASE_DIR = BASE_DIR + "/models"
DATA_BASE_DIR = BASE_DIR + "/dataset"
GENERATIONS_BASE_DIR = BASE_DIR + "/generations"

MODELS_TO_USE = [
    'experiment1'
]

SUPPORTED_METRICS = ['rouge', 'meteor', 'perplexity', 'bertscore']


def log_and_print(msg, logger):
    # print(msg)
    logger.info(msg)

def read_test_set(args):
    with open(DATA_BASE_DIR + args.test_file) as f:
        lines = f.readlines()

    inps = []
    targets = []
    deltas = []
    for i, line in enumerate(lines):
        if args.limit > 0 and i == args.limit:
            break
        # remove for memorability
        inp, s1, target, s2, target_delta = line.split('\t')

        inps.append(inp.strip())
        targets.append(target.strip())
        deltas.append(float(target_delta))

    return inps, targets, deltas


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, help='File with train sentences/inps to get switch', default='/experiment1_speeds_30000.txt')
    parser.add_argument('--test_file', type=str, help='File with test sentences/inps', default='/eval.txt')
    parser.add_argument('--train_file', type=str, help='File with test sentences/inps', default='/train.txt')
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

    train_df = pd.read_csv(DATA_BASE_DIR + args.train_file, sep='\t', header=None)
    train_df = train_df[(train_df.iloc[:, 0].str.len() >= 5) & (train_df.iloc[:, 2].str.len() >= 5)]
    train_df.columns = ['input', 'input_value', 'target', 'target_value', 'delta']
    switch = get_default_switch(list(train_df.delta))

    means = defaultdict(float)
    cnts = defaultdict(int)
    for d in train_df.delta:
        means[switch[d]] += d
        cnts[switch[d]] += 1

    means = {k: v / cnts[k] for k, v in means.items()}
    del train_df

    reverse_switch = {v: k for k, v in switch.items()}

    inps, targets, deltas = read_test_set(args)
    
    # with open(GENERATIONS_BASE_DIR + args.pred_file) as f:
    #     pred_speeds = f.readlines()

    with open(GENERATIONS_BASE_DIR + args.pred_file, 'rb') as f:
        data = f.read()

    pred_speeds = data.decode('utf-8').split('\n')[:-1]

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

    distance_from_bounds = 0
    distance_from_center = 0
    for pred, ref in zip(generated_deltas, deltas):
        tag = switch[ref]
        lower, upper = reverse_switch[tag]
        if lower > upper:
            lower, upper = upper, lower

        center = means[tag]

        if lower <= pred <= upper:
            # within range
            continue

        distance_from_bounds += min(abs(pred - lower), abs(pred - upper))
        distance_from_center += abs(pred - center)
    
    print(f'Distance from bounds: {distance_from_bounds / len(deltas)}')
    print(f'Distance from center: {distance_from_center / len(deltas)}')



