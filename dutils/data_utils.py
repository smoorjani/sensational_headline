import logging
import pandas as pd

import torch
import torch.utils.data as data

from transformers import AutoTokenizer
# from transformers.models.bart.modeling_bart import shift_tokens_right

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

        if self.switch:
            # when using special token as control
            special_token = self.switch[item["deltas"]]
            item["input_txt"] = special_token + " " + self.x_seq[idx]
        else:
            # when using embedding as control
            item["input_txt"] = self.x_seq[idx]

        item["target_txt"] = self.y_seq[idx]

        return item 

    def __len__(self):
        return len(self.y_seq)

def collate_fn(data):
    item_info = {}
    # turn list of dicts into dict of lists
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    item_info = {key: torch.stack(item) if type(item) == list else item for key, item in item_info.items()}

    item_info["deltas"] = torch.FloatTensor(item_info["deltas"])
    # if USE_CUDA:
        # deltas = deltas.to("cuda")

    return item_info 

def get_seq(data, max_len, switch):
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
    
    return dataset

def read_langs(file_name, thd=0.0):

    data = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            elements = line.strip().split("\t")
            if len(elements) != 5:
                continue
            input_txt, s1, target_txt, s2, delta = elements
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
    train_dataset = get_seq(d_train, max_len, switch)
    logging.info("start get seq for test")
    test_dataset = get_seq(d_test, max_len, switch)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_fn)
 
    return train_dataset, test_dataloader, max_r, switch


def truncate_tokenized(inputs, max_len):
    if inputs['input_ids'].shape[-1] > max_len:
        for key, item in inputs.items():
            batch_size, tokens = item.shape
            new_item = torch.zeros(batch_size, max_len)
            new_item = item[:, :max_len]
            inputs[key] = new_item
    return inputs


class DeltaDataset(data.Dataset):
    def __init__(
        self,
        x_seq,
        y_seq,
        xs_seq,
        ys_seq,
        s_seq,
        tokenizer,
        switch,
        evaluation=False,
        max_source_length=256,
        max_target_length=256,
        padding="max_length", # more efficient on GPU
        ignore_pad_token_for_loss=True
    ):
        self.x_seq = x_seq
        self.xs_seq = xs_seq
        self.y_seq = y_seq
        self.ys_seq = ys_seq
        self.s_seq = s_seq

        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("facebook/bart-base")
        self.switch = switch
        self.evaluation = evaluation

        self.max_source_length = max_source_length if max_source_length else 256
        self.max_target_length = max_target_length if max_target_length else 256
        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def __getitem__(self, idx):

        item = {}
        delta = float(self.s_seq[idx])
        input_speeds = float(self.xs_seq[idx])
        target_speeds = float(self.ys_seq[idx])

        inputs = self.x_seq[idx]
        if self.switch:
            # when using special token as control
            prefix = self.switch[delta]
            if type(prefix) == float:
                print('prefix is float', idx, prefix)
            elif type(self.x_seq[idx]) == float:
                print('x_seq is float', idx, self.x_seq[idx])
            inputs = prefix + " " + self.x_seq[idx]
        
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True) #, return_tensors='pt')

        # Tokenize targets with the `text_target` keyword argument
        targets = self.y_seq[idx]
        labels = self.tokenizer(text_target=targets, max_length=self.max_target_length, padding=self.padding, truncation=True) #, return_tensors='pt')
        # labels = labels["input_ids"]

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            # labels["input_ids"] = [
            #     [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            # ]
            # labels["input_ids"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
            labels["input_ids"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
            # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["labels_attn"] = labels["attention_mask"]

        # model_inputs["decoder_input_ids"] = labels

        # model_inputs = {key: item.squeeze(0) for key, item in model_inputs.items()}

        model_inputs["deltas"] = delta
        model_inputs["input_speeds"] = input_speeds
        model_inputs["target_speeds"] = target_speeds

        model_inputs = {key: torch.tensor(item) for key, item in model_inputs.items()}
        # model_inputs["decoder_input_ids"] = shift_tokens_right(model_inputs["labels"].unsqueeze(0), self.tokenizer.pad_token_id, self.tokenizer.eos_token_id).squeeze(0)

        model_inputs = truncate_tokenized(model_inputs, min(self.max_source_length, self.max_target_length))

        return model_inputs

    def __len__(self):
        return len(self.y_seq)


def get_data(train_file, test_file, batch_size, tokenizer=None, max_len=None, thd=None, switch=None, limit=None, control_method='tag'):
    train_df = pd.read_csv(train_file, sep='\t', header=None, nrows=limit)
    train_df = train_df[(train_df.iloc[:, 0].str.len() >= 5) & (train_df.iloc[:, 2].str.len() >= 5)]
    train_df.columns = ['input', 'input_value', 'target', 'target_value', 'delta']
    test_df = pd.read_csv(test_file, sep='\t', header=None, nrows=limit)
    test_df = test_df[(test_df.iloc[:, 0].str.len() >= 5) & (test_df.iloc[:, 2].str.len() >= 5)]
    test_df.columns = ['input', 'input_value', 'target', 'target_value', 'delta']

    max_r_train = train_df.target.map(str).map(len).max()
    max_r_test = test_df.target.map(str).map(len).max()

    switch = None
    if control_method == 'tag':
        switch = switch if switch else get_default_switch(list(train_df.delta))
        special_tokens = list(switch.values())

        print('Adding special tokens')
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    logging.info("finish loading lang")
    max_r = max(max_r_train, max_r_test) + 1
    
    logging.info("start get seq for train")
    train_dataset = DeltaDataset(
        list(train_df.input),
        list(train_df.target),
        list(train_df.input_value),
        list(train_df.target_value),
        list(train_df.delta),
        tokenizer,
        switch,
        max_source_length=max_len,
        max_target_length=max_len,
    )
    
    logging.info("start get seq for test")
    test_dataset = DeltaDataset(
        list(test_df.input)[:1000],
        list(test_df.target)[:1000],
        list(train_df.input_value)[:1000],
        list(train_df.target_value)[:1000],
        list(test_df.delta)[:1000],
        tokenizer,
        switch,
        max_source_length=max_len,
        max_target_length=max_len,
    )

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_fn)
 
    return train_dataset, test_dataloader, max_r, tokenizer


