from __future__ import unicode_literals, print_function, division
from transformers.generation_logits_process import LogitsProcessorList
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer, BartTokenizer, BartConfig, BartForCausalLM
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from utils.masked_cross_entropy import sequence_mask
from utils.rouge import rouge
from utils.config import *
import numpy as np
from utils.embedding_helper import get_embedding

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-0.02, 0.02)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)


def init_wt_normal(wt):
    wt.data.normal_(std=1e-4)


def init_wt_unif(wt):
    wt.data.uniform_(-0.02, 0.02)


class ReduceState(nn.Module):
    def __init__(self, args):
        super(ReduceState, self).__init__()

        self.args = args
        self.reduce_h = nn.Linear(
            self.args["hidden_size"] * 2, self.args["hidden_size"])
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(
            self.args["hidden_size"] * 2, self.args["hidden_size"])
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        hidden_reduced_h = F.relu(self.reduce_h(
            h.view(-1, self.args["hidden_size"] * 2)))
        hidden_reduced_c = F.relu(self.reduce_c(
            c.view(-1, self.args["hidden_size"] * 2)))

        # h, c dim = 1 x b x hidden_dim
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()

        self.args = args
        # attention
        self.W_h = nn.Linear(args["hidden_size"] * 2,
                             args["hidden_size"] * 2, bias=False)
        if self.args["is_coverage"]:
            self.W_c = nn.Linear(1, args["hidden_size"] * 2, bias=False)
        self.decode_proj = nn.Linear(
            args["hidden_size"] * 2, args["hidden_size"] * 2)
        self.v = nn.Linear(args["hidden_size"] * 2, 1, bias=False)

    def forward(self, s_t_hat, h, enc_padding_mask, coverage):
        b, t_k, n = list(h.size())
        h = h.view(-1, n)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(h)

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(
            b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        # B * t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if self.args["is_coverage"]:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(
                coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        h = h.view(-1, t_k, n)  # B x t_k x 2*hidden_dim
        c_t = torch.bmm(attn_dist, h)  # B x 1 x n
        c_t = c_t.view(-1, self.args["hidden_size"] * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.args["is_coverage"]:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


def share_embedding(opts, lang):
    embedding = nn.Embedding(
        lang.n_words, opts['emb_size'], padding_idx=PAD_idx)
    embedding.weight.data.requires_grad = True

    if opts['emb_file'] is not None:
        pre_embedding = get_embedding(
            opts['emb_file'], lang, opts["embedding_key"])
        embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.weight.data.requires_grad = True

    return embedding


class PointerAttnSeqToSeq(nn.Module):

    def __init__(self, args, lang):
        super(PointerAttnSeqToSeq, self).__init__()
        self.args = args
        self.lang = lang
        self.vocab_size = lang.n_words

        self.embedding = share_embedding(self.args, lang)

        print('Loading gpt model...')
        config = GPT2Config.from_pretrained(
            "gpt2", output_hidden_states=True, output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config.pad_token_id = config.eos_token_id
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

        # config = BartConfig.from_pretrained("facebook/bart-base", output_hidden_states=True, output_attentions=True)
        # self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # config.pad_token_id = config.eos_token_id
        # self.decoder = BartForCausalLM.from_pretrained("facebook/bart-base", config=config)

        self.reduce_state = ReduceState(self.args)

        if USE_CUDA:
            self.decoder = self.decoder.to("cuda:0")
            self.embedding = self.embedding.to("cuda:0")
            self.reduce_state = self.reduce_state.to("cuda:0")
        print(f'gpt model on {self.decoder.device}')

    def get_output_from_batch(self, batch):

        dec_batch = batch["target_batch"].transpose(0, 1)
        target_batch = batch["target_ext_vocab_batch"].transpose(0, 1)
        dec_lens_var = batch["target_lengths"]
        max_dec_len = max(dec_lens_var)

        assert max_dec_len == target_batch.size(1)

        dec_padding_mask = sequence_mask(
            dec_lens_var, max_len=max_dec_len).float()

        return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

    def decoded_batch_to_txt(self, all_targets):

        batch_size = all_targets[0].size(0)
        hyp = []
        for i in range(batch_size):
            hyp.append([self.tokenizer.decode(t[i]) for t in all_targets])
        return hyp

    def get_sensation_reward(self, decoded_sents, batch, sensation_model):
        '''
        Gets R_sen
        '''

        joined_decoded_sents = [' '.join(sent) for sent in decoded_sents]
        sents = [pred + ' [SEP] ' + target for pred,
                 target in zip(joined_decoded_sents, batch['target_txt'])]
        batch = self.bert_tokenizer(
            sents, return_tensors='pt', padding=True)['input_ids']

        if USE_CUDA:
            batch = batch.to('cuda:1')

        try:
            rewards = sensation_model(batch)
        except RuntimeError:
            print('Runtime Error!')
            print(f'decoded: {decoded_sents}')
            print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
            raise RuntimeError

        rewards = rewards.to('cuda:0')
        w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list)
                              if len(word_list) else 1 for word_list in decoded_sents])
        if USE_CUDA:
            w = w.to("cuda:0")

        return rewards * w

    def get_reward(self, decoded_sents, batch, sensation_model):

        sensation_rewards = self.get_sensation_reward(
            decoded_sents, batch, sensation_model)
        rewards = sensation_rewards

        rewards = rewards.detach()
        return rewards, 0.0

    def init_batch(self, batch, individual_tokenization=False):
        texts = batch['input_txt']
        target_texts = batch['target_txt']

        inputs, targets, batch_size = None, None, 0
        if individual_tokenization:
            inputs = [self.tokenizer(text, return_tensors="pt")
                      for text in texts]
            targets = [self.tokenizer(target, return_tensors="pt")
                       for target in target_texts]

            if USE_CUDA:
                inputs = [{key: item.to('cuda:0')
                           for key, item in inp.items()} for inp in inputs]
                inputs = [
                    {key: item.to('cuda:0') for key, item in target.items()} for target in targets]

            batch_size = len(inputs)
        else:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

            self.tokenizer.padding_side = "right"
            targets = self.tokenizer(
                target_texts, return_tensors="pt", padding=True)

            if USE_CUDA:
                inputs = {key: item.to('cuda:0')
                          for key, item in inputs.items()}
                targets = {key: item.to('cuda:0')
                           for key, item in targets.items()}

            batch_size = inputs['input_ids'].shape[0]
        # padding should stay on the right for targets
        # this ensures it doesn't interfere with teacher forcing
        return inputs, targets, batch_size

    def run_decoder(self, inputs):
        attention_mask = inputs['attention_mask']
        outputs = self.decoder(**inputs)

        # get next token
        final_dist = outputs.logits[:, -1, :]
        logits_processor = LogitsProcessorList()
        next_tokens_scores = logits_processor(inputs['input_ids'], final_dist)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # appending next tokens to input_ids
        input_ids = torch.cat(
            (inputs['input_ids'], next_tokens.unsqueeze(0).t()), dim=1)

        # checks if last generated token is EOS, and if so, accordingly updates attention mask
        generated_attention_mask = torch.logical_not(
            (inputs['input_ids'][:, -1].clone().detach() == self.tokenizer.eos_token_id)).long().view(-1, 1)
        attention_mask = torch.cat(
            (attention_mask, generated_attention_mask), dim=1)

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return inputs, outputs, final_dist

    def get_rl_loss(self, batch, sensation_model, use_s_score):
        inputs, _, batch_size = self.init_batch(batch)

        step_mask = Variable(torch.ones(batch_size)).float()
        
        if USE_CUDA:
            step_mask = step_mask.to("cuda:0")

        all_step_mask = []
        all_targets = []
        all_output1 = []
        step_losses = []

        # in len of maximum size of headlines
        for di in range(self.args["max_r"]):
            inputs, outputs, final_dist = self.run_decoder(inputs)
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
            step_loss = -torch.log(gold_probs + self.args["eps"])

            step_loss = step_loss * step_mask
            all_step_mask.append(step_mask)
            step_losses.append(step_loss)
            step_mask = torch.clamp(
                step_mask - (target == EOS_idx).float(), min=0.0)

        # this is the linear layer that calculates \hat{R}_t
        baseline_rewards = [self.expected_reward_layer(output1.detach()) * step_mask.unsqueeze(1).detach()
                            for output1, step_mask in zip(all_output1, all_step_mask)]
        baseline_rewards = torch.cat(baseline_rewards, dim=1)
        all_step_mask = torch.stack(all_step_mask, dim=1).float()
        dec_lens_var = torch.sum(all_step_mask, dim=1)
        decoded_sents = self.decoded_batch_to_txt(all_targets)
        total_reward, probs = self.get_reward(
            decoded_sents, batch, sensation_model)
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
        _, ml_loss, _ = self.get_loss(batch, use_s_score=use_s_score)
        if use_s_score:
            loss = rl_loss + ml_loss
        else:
            loss = (1 - self.args["ml_wt"]) * rl_loss + \
                self.args["ml_wt"] * ml_loss

        rewards_loss = torch.sum(
            (total_reward - baseline_rewards) ** 2 * all_step_mask) / torch.sum(all_step_mask)

        return total_reward.mean(), loss, Variable(torch.FloatTensor([0.0])), rewards_loss, probs

    def get_loss(self, batch, use_s_score=False, return_full_loss=False):
        # calculates MLE loss
        # seems like target and dec batches are the same
        _, dec_padding_mask, _, dec_lens_var, target_batch = self.get_output_from_batch(
            batch)

        inputs, targets, _ = self.init_batch(batch)
        step_losses = []

        for di in range(min(targets['input_ids'].shape[-1], self.args["max_r"])):
            inputs, _, final_dist = self.run_decoder(inputs)
            target = target_batch[:, di]
            gold_probs = torch.gather(
                final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.args["eps"])
            
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
            # TODO: how to do teacher forcing here? I could try setting hidden state
            #       but once again, the shapes differ
            # TODO: forces to be teacher's answer and calculates loss based students? what is the teacher and why is this not performing

            # Teacher forcing
            inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        if use_s_score:
            batch_avg_loss = sum_losses / \
                dec_lens_var.float()*batch["sensation_scores"]
        else:
            batch_avg_loss = sum_losses/dec_lens_var.float()
        loss = torch.mean(batch_avg_loss)

        if return_full_loss:
            return None, loss, Variable(torch.FloatTensor([0.0])), batch_avg_loss
        else:
            return None, loss, Variable(torch.FloatTensor([0.0]))

    def get_prob(self, batch):

        _, dec_padding_mask, _, dec_lens_var, target_batch = self.get_output_from_batch(
            batch)

        inputs, targets, _ = self.init_batch(batch)
        step_losses = []

        for di in range(min(targets['input_ids'].shape[-1], self.args["max_r"])):
            inputs, _, final_dist = self.run_decoder(inputs)

            target = target_batch[:, di]

            gold_probs = torch.gather(
                final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.args["eps"])
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

            # Teacher forcing
            inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var.float()
        loss = torch.mean(batch_avg_loss)

        return loss

    def format_output(self, decoder_outputs, batch_size):

        # move all decoder_outputs after EOS as pad
        sampled_length = Variable(torch.zeros(batch_size)).long()
        sampled_outputs = Variable(torch.ones(
            batch_size, self.max_r) * PAD_idx).long()
        if USE_CUDA:
            sampled_outputs = sampled_outputs.to("cuda:0")
            sampled_length = sampled_length.to("cuda:0")
        for i in range(batch_size):
            for j in range(self.max_r):
                sampled_outputs[i, j] = decoder_outputs[j][i]
                if decoder_outputs[j][i] == EOS_idx:
                    sampled_length[i] = j+1
                    break

        return sampled_outputs, sampled_length

    def batch_to_txt(self, input_batch, input_length):

        decoded_sents = []
        for i, input_i in enumerate(input_batch):
            if USE_CUDA:
                decoded_sents.append([self.lang.idx2word[int(ni.cpu().numpy())]
                                     for ni in input_i[:int(input_length[i])]])
            else:
                decoded_sents.append([self.lang.idx2word[int(ni.numpy())]
                                     for ni in input_i[:int(input_length[i])]])
        return decoded_sents

    def decode_batch(self, batch, decode_type):

        self.decoder.train(False)
        assert decode_type == "beam"
        inputs, _, _ = self.init_batch(batch, individual_tokenization=True)
        if USE_CUDA:
            inputs = [{key: item.to('cuda:0')
                       for key, item in inp.items()} for inp in inputs]

        outputs = [self.decoder.generate(
            **inp, num_beams=5, max_length=256) for inp in inputs]

        decoded_sents = [self.tokenizer.decode(
            output[0]) for output in outputs]

        self.decoder.train(True)
        return decoded_sents

    def get_rep_rate(self, sents):
        num_uni_tokens, num_tokens = 0, 0
        for sent in sents:
            tokens = sent.strip().split()
            num_uni_tokens += len(set(tokens))
            num_tokens += len(tokens)
        return 1. - num_uni_tokens * 1.0 / num_tokens

    def evaluate(self, dev, decode_type, return_pred=False, sensation_model=None):

        logging.info("start evaluation")
        hyp = []
        ref = []
        tmp_loss = []
        rewards = []
        sensation_scores = []
        articles = []

        for _, data_dev in enumerate(dev):
            l = self.get_prob(data_dev)
            tmp_loss.append(float(l.data.cpu().numpy()))

            decoded_sents = self.decode_batch(data_dev, decode_type)
            for i, sent in enumerate(decoded_sents):
                hyp.append(" ".join(sent))
                ref.append(" ".join(data_dev["target_txt"][i].split()))
                articles.append(data_dev["input_txt"][i])
            if sensation_model is not None:
                rewards.extend([r for r in self.get_reward(
                    decoded_sents, data_dev, sensation_model)[0]])
            if "sensation_scores" in data_dev:
                sensation_scores.extend([float(s)
                                        for s in data_dev["sensation_scores"]])

        rouge_score = rouge(hyp, ref)
        logging.info("decode type: {}, score: {}, ref repeatition rate: {}, prediction repeatition rate: {}".format(
            decode_type, rouge_score, self.get_rep_rate(ref), self.get_rep_rate(hyp)))
        dev_loss = np.mean(tmp_loss)
        logging.info("dev loss: "+str(dev_loss))
        logging.info("rewards: "+str(sum(rewards) / len(rewards)))

        if return_pred:
            return float(sum(rewards) / len(rewards)), dev_loss, (hyp, ref, rewards, sensation_scores, articles)
        else:
            return float(sum(rewards) / len(rewards)), dev_loss

    def predict_batch(self, batch, decode_type):
        hyp, ref = [], []
        decoded_sents = self.decode_batch(batch, decode_type)
        for i, sent in enumerate(decoded_sents):
            hyp.append(" ".join(sent))
            ref.append(batch["target_txt"][i])
        return hyp, ref
