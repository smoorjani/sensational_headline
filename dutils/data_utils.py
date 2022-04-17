import torch
import torch.utils.data as data
from torch.autograd import Variable
from dutils.config import *
import logging
from transformers import BertTokenizer

class Dataset(data.Dataset):
    def __init__(self, x_seq, y_seq, s_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.s_seq = s_seq
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):

        item = {}
        item["input_txt"] = self.x_seq[idx]
        item["target_txt"] = self.y_seq[idx]
        item["sensation_score"] = self.s_seq[idx]

        return item 

    def __len__(self):
        return len(self.y_seq)

def collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    sensation_scores = Variable(torch.FloatTensor(item_info["sensation_score"]))
    if USE_CUDA:
        sensation_scores = sensation_scores.to("cuda")

    d = {}
    d["input_txt"] = item_info["input_txt"]
    d["target_txt"] = item_info["target_txt"]
    d["sensation_scores"] = sensation_scores

    return d 

def get_seq(data, batch_size, max_len, shuffle=True):
    x_seq, y_seq = [], []
    s_seq = []
    if max_len is not None:
        data = data[:max_len]
    data = data
    for d in data:
        x_seq.append(d["x"])
        y_seq.append(d["y"])
        s_seq.append(d["s"]) 
    
    dataset = Dataset(x_seq, y_seq, s_seq)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
        shuffle=shuffle, collate_fn=collate_fn)
    
    return data_loader

def read_langs(file_name, thd=0.0):

    data = []
    with open(file_name, "r") as f:
            for line in f.readlines():
                elements = line.strip().split("\t")
                if len(elements) != 3:
                    continue
                headline, score, article = elements
                d = {}
                d["x"] = article
                d["y"] = headline
                d["s"] = float(score)
                if d["s"] < thd:
                    continue

                d["x_len"] = len(d["x"].strip().split())
                d["y_len"] = len(d["y"].strip().split())
                data.append(d)

    max_r = max([d["y_len"] for d in data])
    return data, max_r

def prepare_data_seq(train_file, test_file, batch_size, shuffle=True, thd=None):

    file_train = train_file
    file_test = test_file
    logging.info(thd)        
    d_train, max_r_train = read_langs(file_train, thd)
    d_test, max_r_test = read_langs(file_test)
    
    logging.info("finish loading lang")
    max_r = max(max_r_train, max_r_test) + 1
    
    logging.info("start get seq for train")
    max_len = None
    train = get_seq(d_train, batch_size, max_len, shuffle=shuffle)
    logging.info("start get seq for test")
    test = get_seq(d_test, batch_size, max_len, shuffle=False)
 
    return train, test, max_r

