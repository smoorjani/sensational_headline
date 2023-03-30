import torch
import torch.nn as nn

from transformers import BertModel
from nltk import word_tokenize

from dutils.config import *

def tokenize_sents(
    sents,
    tokenizer,
    padding='max_length',
    max_length=512,
    truncation=True,
    is_split_into_words=True,
    device=None,
):
    if device is None:
        device = "cuda"

    batch = tokenizer(
        sents, return_tensors='pt', padding=padding, truncation=truncation,
        is_split_into_words=is_split_into_words, max_length=max_length
    )
    batch = {key: item.to(device) for key, item in batch.items()}

    return batch

def get_discriminator_reward(
    discriminator,
    tokenizer,
    decoded_sents,
    target_deltas,
    precomputed=False,
    input_speeds=None,
    input_sents=None,
    strict_avg=False,
    device=None
):
    assert (precomputed and input_speeds is not None) or (not precomputed and input_sents is None)

    if device is None:
        device = "cuda"
    staging_device = next(discriminator.parameters()).device
    rewards = None

    if not strict_avg:
        input_speeds = input_speeds
        if not precomputed:
            input_speeds = discriminator(
                **tokenize_sents(input_sents, tokenizer, device=staging_device)
            )
        
        generated_speeds = discriminator(
            **tokenize_sents(decoded_sents, tokenizer, device=staging_device)
        )

        if type(generated_speeds) != torch.Tensor:
            generated_speeds = generated_speeds.logits.squeeze(1)
        if type(input_speeds) != torch.Tensor:
            input_speeds = input_speeds.logits.squeeze(1)

        rewards = (generated_speeds - input_speeds) - target_deltas
    else:
        input_speeds = input_speeds
        assert precomputed, "Must be precomputed if strict_avg is True"

        generated_speeds = []
        generated_tokens = word_tokenize(decoded_sents)

        rewards = []

        for i, sentence in enumerate(generated_tokens):
            gen_speeds = discriminator(
                **tokenize_sents(sentence, tokenizer, device=staging_device)
            )

            if type(gen_speeds) != torch.Tensor:
                gen_speeds = gen_speeds.logits.squeeze(1)

            # get diffs
            # [s2 s3 s4 ... sn] - [s1 s2 s3 ... sn-1]
            gen_deltas = gen_speeds[1:] - gen_speeds[:-1]
            gen_deltas -= target_deltas[i]

            reward_i = torch.norm(gen_speeds) / (len(sentence) - 1) # || \sum_i (d_i' - d_i) - delta || / N
            rewards.append(reward_i)

        rewards = torch.stack(rewards)

    rewards = rewards.to(device)
    return rewards


def get_computed_reward(decoded_sents, speeds, deltas, compute_fn, device=None, kwargs=None):
    if device is None:
        device = "cuda"

    generated_rewards = torch.tensor(compute_fn(decoded_sents, **kwargs)).to(device)
    rewards = (generated_rewards - speeds) - deltas

    return rewards.detach()


# def get_discriminator_reward(
#     decoded_sents,
#     target_sents,
#     target_speeds,
#     target_deltas,
#     discriminator,
#     tokenizer,
#     precomputed=False,
#     device=None
# ):
#     '''
#     Gets R_sen
#     '''

#     if device is None:
#         device = "cuda"

#     generated_batch = tokenizer(
#         decoded_sents, return_tensors='pt', padding='max_length', truncation=True,
#         is_split_into_words=True, max_length=512
#     )

#     target_batch, target_values = None, None
#     if not precomputed:
#         target_batch = tokenizer(
#             target_sents, return_tensors='pt', padding='max_length', truncation=True,
#             max_length=512
#         )

#     if USE_CUDA:
#         staging_device = next(discriminator.parameters()).device
#         discriminator = discriminator.to(staging_device)
#         generated_batch = {key: item.to(staging_device) for key, item in generated_batch.items()}
#         target_batch = {key: item.to(staging_device) for key, item in target_batch.items()}

#     try:

#         generated_speeds = discriminator(generated_batch)
#         if not precomputed:
#             target_values = discriminator(target_batch)
#     except RuntimeError as e:
#         print('Runtime Error!')
#         print(e)
#         print(f'decoded: {decoded_sents}')
#         print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
#         raise RuntimeError

#     # no need for norm here, it is done in get_loss (see `sum_losses`)
#     rewards = None
#     if not precomputed:
#         rewards = (generated_speeds - target_values) - target_deltas
#     else:
#         rewards = (generated_speeds - target_speeds) - target_deltas

#     # ratio of unique words to words. not sure if this is needed
#     # w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list)
#     #                         if len(word_list) else 1 for word_list in decoded_sents])
#     if USE_CUDA:
#         rewards = rewards.to(device)

#     # remove from computational graph, no backprop
#     # TODO: consider allowing backprop and simultaneously training discriminator
#     return rewards

