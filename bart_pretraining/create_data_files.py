import os
import json
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/control_tuning/yelpdata/")
    parser.add_argument('--train_val_split', type=float, default=0.1)
    parser.add_argument('--raw_data', type=str, default="yelpdata_deltas_1M.txt")
    parser.add_argument('--train_file', type=str, default="pretraining_data/train.json")
    parser.add_argument('--val_file', type=str, default="pretraining_data/val.json")

    return parser.parse_args()

def json_format(data):
    return json.dumps({
        'text': data[0],
        'summary': data[2]
    }) + '\n'

def main():
    args = get_args()
    
    raw_data_file = args.raw_data
    data_dir = os.environ['PROJECT'] + args.data_dir
    train_val_split = args.train_val_split

    train_file = args.train_file
    val_file = args.val_file

    with open(data_dir + raw_data_file, 'r') as f:
        raw_data = f.readlines()

    raw_data = list(map(
        lambda x: x.replace('"', "'").strip().split('\t'),
        raw_data
    ))

    num_samples = len(raw_data)
    num_train_samples = int(num_samples * (1 - train_val_split))

    random.shuffle(raw_data)
    train_data = raw_data[:num_train_samples]
    val_data = raw_data[num_train_samples:]

    with open(data_dir + train_file, 'w') as f:
        for i in range(len(train_data)):
            f.write(json_format(train_data[i]))

    with open(data_dir + "pretraining_data/train.txt", 'w') as f:
        for i in range(len(train_data)):
            f.write('\t'.join(train_data[i]) + '\n')

    with open(data_dir + val_file, 'w') as f:
        for i in range(len(val_data)):
            f.write(json_format(val_data[i]))
    
    with open(data_dir + "pretraining_data/val.txt", 'w') as f:
        for i in range(len(train_data)):
            f.write('\t'.join(train_data[i]) + '\n')

if __name__ == "__main__":
    main()
