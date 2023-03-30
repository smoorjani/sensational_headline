import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from dutils.data_utils import truncate_tokenized
from dutils.batch_utils import decoded_batch_to_txt, get_output_from_batch, init_batch, run_decoder
from dutils.sensation_scorer import get_computed_reward, get_discriminator_reward
from dutils.masked_cross_entropy import sequence_mask
from dutils.config import *

from dutils.reward_utils.speed import compute_speeds

def get_tuning_loss(args, batch, decoder, tokenizer, discriminator_utils, direct_comp_utils):
    device = torch.device('cuda', args.local_rank)
    batch_size = batch['target_speeds'].shape[0]

    deltas = batch['deltas']
    inputs = {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask']
    }

    outputs = None
    if args.control_method == 'emb':
        outputs = decoder.generate(
            **inputs, 
            min_length=args.max_len // 8,
            max_length=args.max_len, 
            num_beams=1, 
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            control=deltas # handled by model_specific_kwargs
        )
    else:
        outputs = decoder.generate(
            **inputs, 
            min_length=args.max_len // 8,
            max_length=args.max_len, 
            num_beams=1, 
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if random.randint(0, 100) < 5:
        print(decoded_sents)
    
    # targets = []
    # for i in range(batch_size):
    #     # target = batch['labels'][i]
    #     target = batch['input_ids'][i]
    #     target[target == -100] = tokenizer.pad_token_id
    #     targets.append(target)

    # target_sents = tokenizer.batch_decode(targets, skip_special_tokens=True)

    reward = None
    if args.use_discriminator:
        discriminator, classifier_tokenizer = discriminator_utils
        reward = get_discriminator_reward(
            discriminator,
            classifier_tokenizer,
            decoded_sents,
            batch['deltas'],
            precomputed=True,
            input_speeds=batch['input_speeds'],
            strict_avg=args.use_strict_avg,
            device=device
        )
        
    else:
        ft_model, stemmer, en_stop = direct_comp_utils
        reward = get_computed_reward(
            decoded_sents,
            batch['input_speeds'],
            batch['deltas'],
            compute_speeds,
            kwargs={
                'ft_model': ft_model,
                'stemmer': stemmer,
                'en_stop': en_stop
            }
        )

    # mean squared rewards
    return torch.norm(reward), outputs


    # outputs = None
    # enc_output = decoder(**inputs, output_hidden_states=True)
    # del inputs['attention_mask']

    # inputs['decoder_input_ids'] = torch.full((batch_size,1), tokenizer.bos_token_id).to(device)
    # inputs['encoder_outputs'] = enc_output.encoder_last_hidden_state
    # for i in range(len(inputs['encoder_outputs'])):
    #     inputs['encoder_outputs'][i] = inputs['encoder_outputs'][i].to(device)

    # for _ in range(args.max_len):
    #     output = decoder(**inputs, output_hidden_states=True)

    #     final_dist = output.logits[:, -1, :]
        
    #     next_token = None
    #     if sampling == 'greedy':
    #         next_token_scores = F.log_softmax(final_dist, dim=-1)
    #         next_token = torch.argmax(next_token_scores, dim=-1)
    #     else:
    #         filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    #         next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

    #     inputs["decoder_input_ids"] = torch.cat(
    #         (
    #             inputs["decoder_input_ids"],
    #             next_token.unsqueeze(0).t()
    #         ),
    #         dim=1
    #     )
    #     outputs = inputs["decoder_input_ids"]

    #     if torch.all(torch.any(inputs["decoder_input_ids"] == tokenizer.eos_token_id, 1)):
    #         break

def get_rl_loss(args, batch, decoder, tokenizer, discriminator, classifier_tokenizer):
    # print('Running RL loss...')
    device = torch.device('cuda', args.local_rank)
    # batch = {key: item.to(device) for key, item in batch.items()}
    # batch = {key: torch.stack(item).to(device) if type(item) == list else item.to(device) for key, item in batch.items()}
    # inputs = {key: item.clone().detach() for key, item in batch.items()}

    batch_size = batch['deltas'].shape[0]
    step_mask = torch.ones(batch_size).float()
    
    if USE_CUDA:
        step_mask = step_mask.to(device)

    all_step_mask = []
    all_targets = []
    step_losses = []

    inputs = {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask']
    }

    # in len of maximum size
    for di in range(args.max_r):
        inputs, outputs, final_dist, target, _ = run_decoder(decoder, tokenizer, inputs)
        # do this to avoid negatives being fed into multinomial
        final_dist = final_dist.softmax(dim=1)
        
        all_targets.append(target.detach())

        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()  
        step_loss = -torch.log(gold_probs + args.eps) # log P(w_t)
        
        step_loss = step_loss * step_mask
        all_step_mask.append(step_mask)
        step_losses.append(step_loss)

        step_mask = torch.clamp(
            step_mask - (target == tokenizer.eos_token_id).float(), min=0.0)

    all_step_mask = torch.stack(all_step_mask, dim=1).float()
    dec_lens_var = torch.sum(all_step_mask, dim=1)
    
    # decoded_sents = decoded_batch_to_txt(tokenizer, all_targets)
    
    targets = []
    for i in range(batch_size):
        target = batch['labels'][i]
        target[target == -100] = tokenizer.pad_token_id
        targets.append(target)

    target_sents = tokenizer.batch_decode(targets, skip_special_tokens=True)
    decoded_sents = tokenizer.batch_decode(all_targets, skip_special_tokens=True)

    print(target_sents, decoded_sents)

    reward = get_reward(
        decoded_sents, target_sents, batch['deltas'], discriminator, classifier_tokenizer, device)

    # multiply by reward
    sum_losses = torch.sum(reward.squeeze(0) * torch.stack(step_losses, 1), 1)

    batch_avg_loss = sum_losses/dec_lens_var.float()
    value_mse_loss = torch.mean(batch_avg_loss)

    return args.gamma * value_mse_loss

def get_mle_loss(args, batch, decoder, tokenizer):
    # print('Running MLE loss...')
    # calculates MLE loss
    device = torch.device('cuda', args.local_rank)
    batch = {key: item.to(device) for key, item in batch.items()}
    # batch = {key: torch.stack(item).to(device) if type(item) == list else item.to(device) for key, item in batch.items()}
    # inputs = {key: item.clone().detach() for key, item in batch.items()}

    dec_lens_var = torch.count_nonzero(batch['labels_attn'], dim=1)
    max_dec_len = max(dec_lens_var)

    # print(target_batch.size(1), dec_lens_var, max_dec_len)
    # assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(
        dec_lens_var, max_len=batch['labels'].size(1)).float()

    step_losses = []

    for di in range(min(batch['labels'].shape[-1], args.max_r)):
        # step loss is language modelling loss, computed over multiple steps
        inputs, _, _, _, step_loss = run_decoder(decoder, tokenizer, inputs, labels=inputs['input_ids'])

        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

        # Teacher forcing
        inputs['input_ids'][:, -1] = batch['labels'][:, di]

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    batch_avg_loss = sum_losses/dec_lens_var.float()

    return torch.mean(batch_avg_loss)

# def get_rl_loss(args, batch, decoder, tokenizer, discriminator, classifier_tokenizer, expected_reward_layer, tfidf_map, use_s_score, hidden_size=1024):
#     print('Running RL loss...')
#     device = torch.device('cuda', args.local_rank)
#     inputs, _, batch_size = init_batch(tokenizer, batch, device=device)

#     step_mask = Variable(torch.ones(batch_size)).float()
    
#     if USE_CUDA:
#         step_mask = step_mask.to(device)

#     all_step_mask = []
#     all_targets = []
#     all_output1 = []
#     step_losses = []

#     # in len of maximum size of headlines
#     for di in range(args.max_r):
#         inputs, outputs, final_dist, target, _ = run_decoder(decoder, tokenizer, inputs)
#         # do this to avoid negatives being fed into multinomial
#         final_dist = final_dist.softmax(dim=1)
        
#         all_targets.append(target.detach())
#         output1 = outputs['hidden_states'][-1][:, -1, :].to(device).float()
#         # print(output1.shape)
#         # this is some hidden state (batch * hidden_dim) -> o_t
#         all_output1.append(output1)
#         # gold_probs = final_dist[:, target]
#         gold_probs = torch.gather(
#             final_dist, 1, target.unsqueeze(1)).squeeze()
#         step_loss = -torch.log(gold_probs + args.eps)
        
#         step_loss = step_loss * step_mask
#         # print(f'gold_probs {gold_probs.data}\nstep_loss {step_loss.data}\nstep_mask {step_mask.data}\n')
#         all_step_mask.append(step_mask)
#         step_losses.append(step_loss)
#         step_mask = torch.clamp(
#             step_mask - (target == tokenizer.eos_token_id).float(), min=0.0)

#     # this is the linear layer that calculates \hat{R}_t
#     baseline_rewards = [expected_reward_layer(output1.detach()) * step_mask.unsqueeze(1).detach()
#                         for output1, step_mask in zip(all_output1, all_step_mask)]
#                         # batch size x decoding steps

#     baseline_rewards = torch.cat(baseline_rewards, dim=1)
#     all_step_mask = torch.stack(all_step_mask, dim=1).float()
#     dec_lens_var = torch.sum(all_step_mask, dim=1)
    
#     decoded_sents = decoded_batch_to_txt(tokenizer, all_targets)
#     reward = get_reward(
#         decoded_sents, batch['target_txt'], batcha['deltas'], discriminator, classifier_tokenizer, device)
#         # batch_size
#     reward = reward.unsqueeze(1)
    
#     # getting (R - \hat{R}_t)
#     # torch.Size([batch, 1]) torch.Size([batch, max_r])
#     # print(f'sensation {batch["deltas"]}\ntotal_reward {total_reward}\nbaseline_reward {baseline_rewards}')
#     # reward = torch.abs(total_reward.detach() - baseline_rewards.detach())
#     sum_losses = torch.sum(reward * torch.stack(step_losses, 1), 1)
#     # print(f'sensation {(1 - batch["deltas"])}')
#     # print(f'reward: {reward}\nsum: {torch.sum(torch.stack(step_losses, 1), 1)}\nsum_losses {sum_losses}\ndec_lens_var {dec_lens_var}')
#     # this is for ARL
#     if use_s_score:
#         # use the model instead
#         # batch_avg_loss = sum_losses / \
#         #     dec_lens_var.float()*(1 - batch["deltas"])
#         batch_avg_loss = sum_losses / \
#             dec_lens_var.float()*((1 - batch["deltas"]) * args.beta)
#     else:
#         batch_avg_loss = sum_losses/dec_lens_var.float()

#     rl_loss = torch.mean(batch_avg_loss)
#     ml_loss = get_mle_loss(args, batch, decoder, tokenizer)

#     print(f'rl_loss: {rl_loss}, ml_loss: {ml_loss}')

#     if use_s_score:
#         loss = rl_loss + ml_loss
#     else:
#         loss = (1 - args.ml_wt) * rl_loss + \
#             args.ml_wt * ml_loss

#     rewards_loss = torch.sum(
#         (total_reward - baseline_rewards) ** 2 * all_step_mask) / torch.sum(all_step_mask)

#     return total_reward.mean(), loss, rewards_loss

# def get_supervised_loss(args, batch, decoder, tokenizer, discriminator, classifier_tokenizer, tfidf_map, use_s_score):
#     print('Running Supervised loss...')
#     device = torch.device('cuda', args.local_rank)
#     inputs, targets, batch_size = init_batch(tokenizer, batch, device=device)
#     target_batch, dec_padding_mask, _, dec_lens_var = get_output_from_batch(
#         targets)

#     step_losses = []
#     all_targets = []

#     step_mask = Variable(torch.ones(batch_size)).float()
    
#     if USE_CUDA:
#         step_mask = step_mask.to(device)

#     # in len of maximum size of headlines
#     for di in range(args.max_r):
#         inputs, outputs, final_dist, target, _ = run_decoder(decoder, tokenizer, inputs)
#         all_targets.append(target.detach())
#         # print(inputs['input_ids'], inputs['input_ids'].shape)
#         decoded_sents = decoded_batch_to_txt(tokenizer, all_targets)
        
#         # decoded_sents = list(map(lambda x: x.split(), tokenizer.batch_decode(inputs['input_ids'])))
#         if di == 0:
#             step_loss = torch.zeros(batch_size, device=device)
#         else:
#             step_loss = get_reward(decoded_sents, batch['target_txt'], discriminator, classifier_tokenizer, device)
#         step_loss = step_loss.unsqueeze(1)
        
#         step_loss = step_loss * step_mask
#         step_losses.append(step_loss)
#         step_mask = torch.clamp(
#             step_mask - (target == tokenizer.eos_token_id).float(), min=0.0)
    
#     sum_losses = torch.sum(torch.stack(step_losses, 1), 1)

#     # this is for ARL
#     if use_s_score:
#         # use the model instead
#         batch_avg_loss = sum_losses / \
#             dec_lens_var.float()*((1 - batch["deltas"]) * args.beta)
#     else:
#         batch_avg_loss = sum_losses/dec_lens_var.float()

#     supervised_loss = torch.mean(batch_avg_loss)
#     ml_loss = get_mle_loss(args, batch, decoder, tokenizer)

#     if use_s_score:
#         print('Using AP loss...')
#         loss = supervised_loss + ml_loss
#     else:
#         loss = (1 - args.ml_wt) * supervised_loss + \
#             args.ml_wt * ml_loss

#     return loss

def get_prob(args, decoder, tokenizer, batch):
    device = torch.device('cuda', args.local_rank)
    inputs, targets, _ = init_batch(tokenizer, batch, device=device)
    target_batch, dec_padding_mask, _, dec_lens_var = get_output_from_batch(
        targets)

    
    step_losses = []

    for di in range(min(targets['input_ids'].shape[-1], args.max_r)):
        inputs, _, final_dist, _, _ = run_decoder(decoder, tokenizer, inputs)

        target = target_batch[:, di]

        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + args.eps)
        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

        # Teacher forcing
        inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    batch_avg_loss = sum_losses/dec_lens_var.float()
    loss = torch.mean(batch_avg_loss)

    return loss