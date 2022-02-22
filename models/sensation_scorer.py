import torch
import torch.nn as nn
from transformers import BertModel

from dutils.config import *

class PersuasivenessClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, dropout=0.2, n_classes=2):
        super(PersuasivenessClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        outputs = self.bert(**input_batch)
        output = self.dropout(outputs.pooler_output)
        output = self.linear(output)
        output = self.softmax(output)
        return torch.max(output)

def get_reward(decoded_sents, target_sents, sensation_model, tokenizer, device=None):
    '''
    Gets R_sen
    '''

    if device is None:
        device = "cuda"

    joined_decoded_sents = [' '.join(sent) for sent in decoded_sents]
    sents = [pred + ' [SEP] ' + target for pred,
                target in zip(joined_decoded_sents, target_sents)]
    batch = tokenizer(
        sents, return_tensors='pt', padding=True)

    if USE_CUDA:
        batch = {key: item.to(device) for key, item in batch.items()}

    try:
        rewards = sensation_model(batch)
    except RuntimeError:
        print('Runtime Error!')
        print(f'decoded: {decoded_sents}')
        print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
        raise RuntimeError

    w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list)
                            if len(word_list) else 1 for word_list in decoded_sents])
    if USE_CUDA:
        rewards = rewards.to(device)
        w = w.to(device)
    sensation_reward = rewards * w
    return sensation_reward.detach()