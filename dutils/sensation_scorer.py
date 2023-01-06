import torch
import torch.nn as nn
from transformers import BertModel

from dutils.config import *

def truncate_batch(batch):
    if batch['input_ids'].shape[-1] > 512:
        for key, item in batch.items():
            batch_size, tokens = item.shape
            new_item = torch.zeros(batch_size, 512)
            new_item = item[:, (tokens-512):]
            batch[key] = new_item
    return batch

def get_reward(decoded_sents, target_sents, deltas, discriminator, tokenizer, device=None):
    '''
    Gets R_sen
    '''

    if device is None:
        device = "cuda"

    generated_batch = truncate_batch(tokenizer(decoded_sents, return_tensors='pt', padding=True))
    target_batch = truncate_batch(tokenizer(target_sents, return_tensors='pt', padding=True))
    

    if USE_CUDA:
        staging_device = next(discriminator.parameters()).device
        discriminator = discriminator.to(staging_device)
        generated_batch = {key: item.to(staging_device) for key, item in generated_batch.items()}
        target_batch = {key: item.to(staging_device) for key, item in target_batch.items()}

    try:
        generated_values = discriminator(generated_batch)
        target_values = discriminator(target_batch)
    except RuntimeError:
        print('Runtime Error!')
        print(f'decoded: {decoded_sents}')
        print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
        raise RuntimeError

    rewards = torch.norm(target_values - generated_values)

    # ratio of unique words to words. not sure if this is needed
    # w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list)
    #                         if len(word_list) else 1 for word_list in decoded_sents])
    if USE_CUDA:
        rewards = rewards.to(device)

    # remove from computational graph, no backprop
    # TODO: consider allowing backprop and simultaneously training discriminator
    return rewards.detach()