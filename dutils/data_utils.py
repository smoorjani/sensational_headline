import logging

import torch
import torch.utils.data as data
from torch.autograd import Variable

from transformers import BertTokenizer
from nltk.corpus import stopwords

from dutils.config import *
from dutils.function import get_default_switch
# from sklearn.feature_extraction.text import TfidfVectorizer

class Dataset(data.Dataset):
    def __init__(self, x_seq, y_seq, s_seq, switch):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.s_seq = s_seq
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.switch = switch

    def __getitem__(self, idx):

        item = {}
        item["deltas"] = float(self.s_seq[idx])
        special_token = self.switch[item["deltas"]]
        item["input_txt"] = special_token + " " + self.x_seq[idx]
        item["target_txt"] = self.y_seq[idx]

        return item 

    def __len__(self):
        return len(self.y_seq)

def collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    deltas = Variable(torch.FloatTensor(item_info["deltas"]))
    # if USE_CUDA:
        # deltas = deltas.to("cuda")

    d = {}
    d["input_txt"] = item_info["input_txt"]
    d["target_txt"] = item_info["target_txt"]
    d["deltas"] = deltas

    return d 

def get_seq(data, batch_size, max_len, switch, shuffle=True):
    x_seq, y_seq = [], []
    s_seq = []
    if max_len is not None:
        data = data[:max_len]
    data = data
    for d in data:
        x_seq.append(d["x"])
        y_seq.append(d["y"])
        s_seq.append(d["s"]) 
    
    dataset = Dataset(x_seq, y_seq, s_seq, switch)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
        shuffle=shuffle, collate_fn=collate_fn)
    
    return dataset, data_loader

def read_langs(file_name, thd=0.0):

    data = []
    with open(file_name, "r") as f:
            for line in f.readlines():
                elements = line.strip().split("\t")
                if len(elements) != 3:
                    continue
                input_txt, target_txt, delta = elements
                d = {}
                d["x"] = target_txt
                d["y"] = input_txt
                d["s"] = float(delta)
                if abs(d["s"]) < thd:
                    continue

                d["x_len"] = len(d["x"].strip().split())
                d["y_len"] = len(d["y"].strip().split())
                data.append(d)

    max_r = max([d["y_len"] for d in data])
    return data, max_r

def prepare_data_seq(train_file, test_file, batch_size, shuffle=True, thd=None, switch=None):

    file_train = train_file
    file_test = test_file
    logging.info(thd)        
    d_train, max_r_train = read_langs(file_train, thd)
    d_test, max_r_test = read_langs(file_test)

    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(articles)
    # tfidf_map = dict(zip(vectorizer.get_feature_names(), X.toarray()[0]))

    # # remove stopwords from tfidf
    # stop_words = set(stopwords.words('english'))
    # tfidf_map = {k:v for k,v in tfidf_map.items() if k not in stop_words}

    switch = switch if switch else get_default_switch([item['s'] for item in d_train])
    
    logging.info("finish loading lang")
    max_r = max(max_r_train, max_r_test) + 1
    
    logging.info("start get seq for train")
    max_len = None
    train_dataset, train = get_seq(d_train, batch_size, max_len, switch, shuffle=shuffle)
    logging.info("start get seq for test")
    _, test = get_seq(d_test, batch_size, max_len, switch, shuffle=False)
 
    return train_dataset, test, max_r, switch

