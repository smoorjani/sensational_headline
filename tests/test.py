from transformers.models.bart_speed.modeling_sbart import SBartForConditionalGeneration
model = SBartForConditionalGeneration.from_pretrained('facebook/bart-base')
print(sum(p.numel() for p in model.parameters()))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
encoding = tokenizer('"We are very happy to show you the 🤗 Transformers library.', return_tensors='pt')

import torch

encoding['control'] = torch.tensor([1.54])
model(**encoding)