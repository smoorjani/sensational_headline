import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argument for nn parameters")
    
    parser.add_argument('--proj_path', type=str, default="/projects/bblr/smoorjani", help="project path")
    parser.add_argument('--raw_data', type=str, default="/control_tuning/yelpdata/yelpdata_deltas_100k.txt", help="raw data") 
    parser.add_argument('--train_data', type=str, default="/control_tuning/dataset/train.txt", help="training data") 
    parser.add_argument('--test_data', type=str, default="/control_tuning/dataset/eval.txt", help="eval data")
    

    parser.add_argument('--train_test_split', type=float, default=0.9, help="ratio of data for training")

    args = parser.parse_args()

    with open(args.proj_path + args.raw_data, 'r') as f:
        raw_data = f.readlines()

    random.shuffle(raw_data)
    cutoff = int(len(raw_data) * args.train_test_split)

    train_data = raw_data[:cutoff]
    test_data = raw_data[cutoff:]

    with open(args.proj_path + args.train_data, 'w') as f:
        for item in train_data:
            f.write(item)

    with open(args.proj_path + args.test_data, 'w') as f:
        for item in test_data:
            f.write(item)