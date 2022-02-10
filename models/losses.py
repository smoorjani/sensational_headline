import torch
from torch.autograd import Variable

from models.batch_utils import decoded_batch_to_txt, get_output_from_batch, init_batch, run_decoder
from models.sensation_scorer import get_reward
from utils.config import *

def get_rl_loss(args, batch, decoder, tokenizer, sensation_model, classifier_tokenizer, expected_reward_layer, use_s_score):
    inputs, _, batch_size = init_batch(tokenizer, batch)

    step_mask = Variable(torch.ones(batch_size)).float()
    
    if USE_CUDA:
        step_mask = step_mask.to("cuda:0")

    all_step_mask = []
    all_targets = []
    all_output1 = []
    step_losses = []

    # in len of maximum size of headlines
    for di in range(args["max_r"]):
        inputs, outputs, final_dist = run_decoder(decoder, tokenizer, inputs)
        # do this to avoid negatives being fed into multinomial
        final_dist = final_dist.softmax(dim=1)
        target = torch.multinomial(
            final_dist.data, 1).long().squeeze()  # sampling
        all_targets.append(target.detach())
        output1 = outputs['hidden_states'][-1][:, -1, :]
        # this is some hidden state (batch * hidden_dim) -> o_t
        all_output1.append(output1)
        # gold_probs = final_dist[:, target]
        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + args["eps"])

        step_loss = step_loss * step_mask
        all_step_mask.append(step_mask)
        step_losses.append(step_loss)
        step_mask = torch.clamp(
            step_mask - (target == tokenizer.eos_token_id).float(), min=0.0)

    # this is the linear layer that calculates \hat{R}_t
    baseline_rewards = [expected_reward_layer(output1.detach()) * step_mask.unsqueeze(1).detach()
                        for output1, step_mask in zip(all_output1, all_step_mask)]
    baseline_rewards = torch.cat(baseline_rewards, dim=1)
    all_step_mask = torch.stack(all_step_mask, dim=1).float()
    dec_lens_var = torch.sum(all_step_mask, dim=1)
    decoded_sents = decoded_batch_to_txt(tokenizer, all_targets)
    total_reward = get_reward(
        classifier_tokenizer, decoded_sents, batch, sensation_model)
    total_reward = total_reward.unsqueeze(1)

    # getting (R - \hat{R}_t)
    # torch.Size([batch, 1]) torch.Size([batch, max_r])
    reward = total_reward.detach() - baseline_rewards.detach()
    sum_losses = torch.sum(reward * torch.stack(step_losses, 1), 1)
    # this is for ARL
    if use_s_score:
        batch_avg_loss = sum_losses / \
            dec_lens_var.float()*(1 - batch["sensation_scores"])
    else:
        batch_avg_loss = sum_losses/dec_lens_var.float()
    rl_loss = torch.mean(batch_avg_loss)
    ml_loss = get_loss(args, decoder, tokenizer, batch, use_s_score=use_s_score)
    if use_s_score:
        loss = rl_loss + ml_loss
    else:
        loss = (1 - args["ml_wt"]) * rl_loss + \
            args["ml_wt"] * ml_loss

    rewards_loss = torch.sum(
        (total_reward - baseline_rewards) ** 2 * all_step_mask) / torch.sum(all_step_mask)

    return total_reward.mean(), loss, rewards_loss

def get_loss(args, decoder, tokenizer, batch, use_s_score=False):
    # calculates MLE loss
    # seems like target and dec batches are the same
    _, dec_padding_mask, _, dec_lens_var, target_batch = get_output_from_batch(
        batch)

    inputs, targets, _ = init_batch(tokenizer, batch)
    step_losses = []

    for di in range(min(targets['input_ids'].shape[-1], args["max_r"])):
        inputs, _, final_dist = run_decoder(decoder, tokenizer, inputs)
        target = target_batch[:, di]
        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + args["eps"])
        
        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

        # Teacher forcing
        inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    if use_s_score:
        batch_avg_loss = sum_losses / \
            dec_lens_var.float()*batch["sensation_scores"]
    else:
        batch_avg_loss = sum_losses/dec_lens_var.float()
    loss = torch.mean(batch_avg_loss)

    return loss

def get_prob(args, decoder, tokenizer, batch):
    _, dec_padding_mask, _, dec_lens_var, target_batch = get_output_from_batch(
        batch)

    inputs, targets, _ = init_batch(tokenizer, batch)
    step_losses = []

    for di in range(min(targets['input_ids'].shape[-1], args["max_r"])):
        inputs, _, final_dist = run_decoder(decoder, tokenizer, inputs)

        target = target_batch[:, di]

        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + args["eps"])
        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

        # Teacher forcing
        inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    batch_avg_loss = sum_losses/dec_lens_var.float()
    loss = torch.mean(batch_avg_loss)

    return loss