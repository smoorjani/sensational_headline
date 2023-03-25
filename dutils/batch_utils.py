from transformers import LogitsProcessorList
import torch
import torch.nn.functional as F

from dutils.masked_cross_entropy import sequence_mask
from dutils.config import *


def init_batch(tokenizer, batch, individual_tokenization=False, device=None, max_len=128):
    if device is None:
        device = "cuda"

    texts = batch['input_txt']
    target_texts = batch['target_txt']

    inputs, targets, batch_size = None, None, 0

    # if individual_tokenization:
    #     inputs = [tokenizer(text, return_tensors="pt")
    #                 for text in texts]
    #     targets = [tokenizer(target, return_tensors="pt")
    #                 for target in target_texts]

    #     if USE_CUDA:
    #         inputs = [{key: item.to(device)
    #                     for key, item in inp.items()} for inp in inputs]
    #         inputs = [
    #             {key: item.to(device) for key, item in target.items()} for target in targets]

    #     batch_size = len(inputs)
    # else:

    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = truncate_tokenized(inputs, max_len)

    tokenizer.padding_side = "right"
    targets = tokenizer(
        target_texts, return_tensors="pt", padding=True)
    # no need to truncate because of sliding window.

    if USE_CUDA:
        inputs = {key: item.to(device)
                    for key, item in inputs.items()}
        targets = {key: item.to(device)
                    for key, item in targets.items()}

    batch_size = inputs['input_ids'].shape[0]

    # padding should stay on the right for targets
    # this ensures it doesn't interfere with teacher forcing
    return inputs, targets, batch_size

def run_decoder(decoder, tokenizer, inputs, limit=64, labels=None):
    # print(inputs['input_ids'].shape)
    attention_mask = inputs['attention_mask']
    if labels == None:
        outputs = decoder(**inputs)
    else:
        outputs = decoder(**inputs, labels=labels)
    # get next token

    final_dist = outputs.logits[:, -1, :]

    logits_processor = decoder._get_logits_processor(
            repetition_penalty = None,
            no_repeat_ngram_size = 3, 
            encoder_no_repeat_ngram_size = None,
            encoder_input_ids = inputs['input_ids'],
            bad_words_ids = None,
            min_length = 8,
            max_length = 64,
            eos_token_id = tokenizer.eos_token_id,
            forced_bos_token_id = None,
            forced_eos_token_id = None,
            prefix_allowed_tokens_fn = None,
            num_beams = 1,
            num_beam_groups = 1,
            diversity_penalty = None,
            remove_invalid_values = None,
            input_ids_seq_length = inputs['input_ids'].shape[-1],
            exponential_decay_length_penalty = None,
            logits_processor = [],
            renormalize_logits = None,
        )
    next_token_logits = logits_processor(inputs['input_ids'], final_dist)
    next_token_scores = F.log_softmax(next_token_logits, dim=-1)

    # next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
    # next_tokens = torch.gather(next_token_scores, -1, _indices)
    # print(next_tokens.shape)

    # TODO(RL Loss): try sampling
    # target = torch.multinomial(
        #     final_dist.data, 1).long().squeeze()  # sampling
    next_tokens = torch.argmax(next_token_scores, dim=-1)

    # appending next tokens to input_ids
    input_ids = torch.cat(
        (inputs['input_ids'], next_tokens.unsqueeze(0).t()), dim=1)

    # checks if last generated token is EOS, and if so, accordingly updates attention mask
    generated_attention_mask = torch.logical_not(
        (inputs['input_ids'][:, -1].clone().detach() == tokenizer.eos_token_id)).long().view(-1, 1)
    attention_mask = torch.cat(
        (attention_mask, generated_attention_mask), dim=1)

    if limit:
        # sliding window over hidden states
        inputs = {'input_ids': input_ids[:, -limit:], 'attention_mask': attention_mask[:, -limit:]}
    else:
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    loss = outputs.loss if labels is not None else 0
    return inputs, outputs, final_dist, next_tokens, loss

def get_output_from_batch(targets):

    target_batch = targets['input_ids']
    dec_lens_var = torch.count_nonzero(targets['attention_mask'], dim=1)
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(
        dec_lens_var, max_len=max_dec_len).float()

    #print(target_batch, dec_padding_mask, max_dec_len, dec_lens_var)
    return target_batch, dec_padding_mask, max_dec_len, dec_lens_var

def decoded_batch_to_txt(tokenizer, all_targets):

    batch_size = all_targets[0].size(0)
    hyp = []
    for i in range(batch_size):
        hyp.append([tokenizer.decode(t[i]) for t in all_targets])
    return hyp

def decode_batch(decoder, tokenizer, batch, device):
    if device is None:
        device = "cuda"

    decoder.train(False)
    inputs, _, _ = init_batch(batch, individual_tokenization=True, device=device)
    if USE_CUDA:
        inputs = [{key: item.to(device)
                    for key, item in inp.items()} for inp in inputs]

    outputs = [decoder.generate(
        **inp, num_beams=5, max_length=256) for inp in inputs]

    decoded_sents = [tokenizer.decode(
        output[0]) for output in outputs]

    decoder.train(True)
    return decoded_sents


