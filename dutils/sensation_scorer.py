import torch
import torch.nn as nn
from transformers import BertModel

from dutils.config import *

def get_discriminator_reward(
    decoded_sents,
    target_sents,
    target_speeds,
    target_deltas,
    discriminator,
    tokenizer,
    precomputed=False,
    device=None
):
    '''
    Gets R_sen
    '''

    if device is None:
        device = "cuda"

    generated_batch = tokenizer(
        decoded_sents, return_tensors='pt', padding='max_length', truncation=True,
        is_split_into_words=True, max_length=512
    )

    target_batch, target_values = None, None
    if not precomputed:
        target_batch = tokenizer(
            target_sents, return_tensors='pt', padding='max_length', truncation=True,
            max_length=512
        )

    if USE_CUDA:
        staging_device = next(discriminator.parameters()).device
        discriminator = discriminator.to(staging_device)
        generated_batch = {key: item.to(staging_device) for key, item in generated_batch.items()}
        target_batch = {key: item.to(staging_device) for key, item in target_batch.items()}

    try:

        generated_values = discriminator(generated_batch)
        if not precomputed:
            target_values = discriminator(target_batch)
    except RuntimeError as e:
        print('Runtime Error!')
        print(e)
        print(f'decoded: {decoded_sents}')
        print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
        raise RuntimeError

    # no need for norm here, it is done in get_loss (see `sum_losses`)
    rewards = None
    if not precomputed:
        rewards = (generated_values - target_values) - target_deltas
    else:
        rewards = (generated_values - target_speeds) - target_deltas

    # ratio of unique words to words. not sure if this is needed
    # w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list)
    #                         if len(word_list) else 1 for word_list in decoded_sents])
    if USE_CUDA:
        rewards = rewards.to(device)

    # remove from computational graph, no backprop
    # TODO: consider allowing backprop and simultaneously training discriminator
    return rewards

def get_computed_reward(decoded_sents, speeds, deltas, compute_fn, device=None, kwargs=None):
    if device is None:
        device = "cuda"

    generated_rewards = torch.tensor(compute_fn(decoded_sents, **kwargs)).to(device)
    rewards = (generated_rewards - speeds) - deltas
    # print(generated_rewards, target_speeds, rewards)

    # if USE_CUDA:
    #     rewards = rewards.to(device)

    return rewards.detach()
