import torch
import torch.utils.data as data
from torch.autograd import Variable
from dutils.config import *
import logging
from transformers import BertTokenizer

class Dataset(data.Dataset):
    def __init__(self, x_seq, y_seq, s_seq, max_q):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.s_seq = s_seq
        self.max_q = max_q
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):

        item = {}
        item["input_txt"] = self.x_seq[idx]
        item["target_txt"] = self.y_seq[idx]
        item["sensation_score"] = self.s_seq[idx]
        item["input_batch"] = self.process(item["input_txt"], False, self.tokenizer)
        item["target_batch"] = self.process(item["target_txt"], True, self.tokenizer)
        item["max_q"] = self.max_q

        return item 

    def __len__(self):
        return len(self.y_seq)

    def process(self, input_txt, target, tokenizer):
        if target:
            seq = tokenizer(input_txt.strip() + " [EOS]")['input_ids']
        else:
            seq = tokenizer(input_txt.strip())['input_ids']
        seq = torch.Tensor(seq)
        return seq

def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        if max_len:
            lengths = [len(seq) if len(seq) < max_len[0] else max_len[0] for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            if max_len:
                end = min(lengths[i], max_len[0])
            else:
                end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x["input_batch"]), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    input_batch, input_lengths = merge(item_info['input_batch'], item_info['max_q'])
    target_batch, target_lengths = merge(item_info['target_batch'], None)
    input_batch = Variable(input_batch).transpose(0, 1)
    target_batch = Variable(target_batch).transpose(0, 1)
    input_lengths = Variable(torch.LongTensor(input_lengths))
    target_lengths = Variable(torch.LongTensor(target_lengths))
    sensation_scores = Variable(torch.FloatTensor(item_info["sensation_score"]))

    if USE_CUDA:
        input_batch = input_batch.to("cuda")
        target_batch = target_batch.to("cuda")
        input_lengths = input_lengths.to("cuda")
        target_lengths = target_lengths.to("cuda")
        sensation_scores = sensation_scores.to("cuda")

    d = {}
    d["input_batch"] = input_batch
    d["target_batch"] = target_batch
    d["input_lengths"] = input_lengths
    d["target_lengths"] = target_lengths
    d["input_txt"] = item_info["input_txt"]
    d["target_txt"] = item_info["target_txt"]
    d["sensation_scores"] = sensation_scores

    return d 

def get_seq(data, batch_size, max_q, max_len, shuffle=True):
    x_seq, y_seq = [], []
    s_seq = []
    if max_len is not None:
        data = data[:max_len]
    data = data[:max_len]
    for d in data:
        x_seq.append(d["x"])
        y_seq.append(d["y"])
        s_seq.append(d["s"]) 
    
    
    dataset = Dataset(x_seq, y_seq, s_seq, max_q)
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

    max_q = max([d["x_len"] for d in data])
    max_r = max([d["y_len"] for d in data])

    return data, max_q, max_r

def prepare_data_seq(train_file, test_file, batch_size, shuffle=True, thd=None):

    file_train = train_file
    file_test = test_file
    logging.info(thd)        
    d_train, max_q_train, max_r_train = read_langs(file_train, thd)
    d_test, max_q_test, max_r_test = read_langs(file_test)
    
    logging.info("finish loading lang")
    
    max_q = max(max_q_train, max_q_test) + 1
    max_r = max(max_r_train, max_r_test) + 1
    max_q = min(max_q, 400)
    
    logging.info("start get seq for train")
    max_len = 384
    train = get_seq(d_train, batch_size, max_q, max_len, shuffle=shuffle)
    logging.info("start get seq for test")
    test = get_seq(d_test, batch_size, max_q, max_len, shuffle=False)
 
    return train, test, max_q, max_r

