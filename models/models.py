import torch
import torch.nn as nn
from transformers import BertModel

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