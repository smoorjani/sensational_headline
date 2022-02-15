from __future__ import unicode_literals, print_function, division
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random
from sutils.masked_cross_entropy import sequence_mask
from sutils.beam_for_pointer_attn import *
from sutils.config import  *
from sutils.utils_sensation import input_txt_to_batch
import numpy as np
from sutils.embedding_helper import get_embedding
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer, BartTokenizer, BartConfig, BartForCausalLM
import deepspeed
from transformers.generation_logits_process import LogitsProcessorList
from transformers.deepspeed import HfDeepSpeedConfig
import gc


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

class BiLSTMEncoder(nn.Module):
    def __init__(self, args, embedding):
        super(BiLSTMEncoder, self).__init__()

        self.args = args
        self.n_layers = args["encoder_layers"]
        self.hidden_size = args["hidden_size"]
        self.lstm = nn.LSTM(args["emb_size"], args["hidden_size"], num_layers=args["encoder_layers"], batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        self.lstm.flatten_parameters()
        self.embedding = embedding

        if self.args["use_oov_emb"]:
            self.oov_emb_proj = nn.Linear(2 * self.hidden_size, self.args["emb_size"])

    def init_hidden(self, input):

        batch_size = input.size(0)
        h0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        c0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if USE_CUDA:
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda()
        return h0_encoder, c0_encoder

    #seq_lens should be in descending order
    def forward(self, enc_input, seq_lens):

        # h0_encoder, c0_encoder = self.init_hidden(embedded)
        # packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        # output, hidden = self.lstm(packed, (h0_encoder, c0_encoder))
        embedded = self.embedding(enc_input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(packed)
        h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
       
        batch_size = enc_input.size(0) 
        if self.args["use_oov_emb"]:
            for i in range(batch_size):
                for  j in range(seq_lens[i]):
                    if enc_input[i, j] == UNK_idx:
                        unk_emb = Variable(torch.zeros(2 * self.hidden_size))
                        if USE_CUDA:
                            unk_emb = unk_emb.cuda()
                        if j > 0:
                            unk_emb[:self.hidden_size] = h[i, j - 1, :self.hidden_size]
                        if j < seq_lens[i] - 1:
                            unk_emb[self.hidden_size:] = h[i, j + 1, self.hidden_size:]
                        embedded[i, j] = self.oov_emb_proj(unk_emb)

            packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
            output, hidden = self.lstm(packed)
            h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n

        h = h.contiguous()

        return h, hidden

class ReduceState(nn.Module):
    def __init__(self, args):
        super(ReduceState, self).__init__()

        self.args = args
        self.reduce_h = nn.Linear(self.args["hidden_size"] * 2, self.args["hidden_size"])
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(self.args["hidden_size"] * 2, self.args["hidden_size"])
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        hidden_reduced_h = F.relu(self.reduce_h(h.view(-1, self.args["hidden_size"] * 2)))
        hidden_reduced_c = F.relu(self.reduce_c(c.view(-1, self.args["hidden_size"] * 2)))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()

        self.args = args
        # attention
        self.W_h = nn.Linear(args["hidden_size"] * 2, args["hidden_size"] * 2, bias=False)
        if self.args["is_coverage"]:
            self.W_c = nn.Linear(1, args["hidden_size"] * 2, bias=False)
        self.decode_proj = nn.Linear(args["hidden_size"] * 2, args["hidden_size"] * 2)
        self.v = nn.Linear(args["hidden_size"] * 2, 1, bias=False)

    def forward(self, s_t_hat, h, enc_padding_mask, coverage):
        b, t_k, n = list(h.size())
        h = h.view(-1, n)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(h)

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if self.args["is_coverage"]:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
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

class PointerAttnDecoder(nn.Module):
    def __init__(self, args, embedding):
        super(PointerAttnDecoder, self).__init__()

        self.args = args
        self.attention_network = Attention(args)
        # decoder
        self.x_context = nn.Linear(self.args["hidden_size"] * 2 + self.args["emb_size"], self.args["emb_size"])

        self.lstm = nn.LSTM(self.args["emb_size"], self.args["hidden_size"], num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)
        self.lstm.flatten_parameters()

        if self.args["pointer_gen"]:
            self.p_gen_linear = nn.Linear(self.args["hidden_size"] * 4 + self.args["emb_size"], 1)
        if self.args["use_oov_emb"]:
            self.oov_proj = nn.Linear(self.args["hidden_size"] * 2, self.args["emb_size"])

        #p_vocab
        self.out1 = nn.Linear(self.args["hidden_size"] * 3, self.args["hidden_size"])
        self.out2 = nn.Linear(self.args["hidden_size"], self.args["vocab_size"])
        init_linear_wt(self.out2)
        self.embedding = embedding

    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step, training):

        if not training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.args["hidden_size"]),
                                 c_decoder.view(-1, self.args["hidden_size"])), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        if self.args["use_oov_emb"]:
            for i in range(y_t_1.size(0)):
                if y_t_1[i] == UNK_idx:
                    y_t_1_embd[i] = self.oov_proj(torch.cat([s_t_1[0][:,i], s_t_1[1][:,i]], dim=-1)).squeeze()
        # =====
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        self.lstm.flatten_parameters()
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        # =====

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.args["hidden_size"]),
                             c_decoder.view(-1, self.args["hidden_size"])), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage)

        if training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.args["pointer_gen"]:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        # =====
        output = torch.cat((lstm_out.view(-1, self.args["hidden_size"]), c_t), 1) # B x hidden_dim * 3
        output1 = self.out1(output) # B x hidden_dim
        # =====

        #output = F.relu(output)

        output = self.out2(output1) # B x vocab_size
        output[:,UNK_idx] = -1e18
        vocab_dist = F.softmax(output, dim=1)

        if self.args["pointer_gen"]:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage, output1


def share_embedding(opts, lang):
        embedding = nn.Embedding(lang.n_words, opts['emb_size'], padding_idx=PAD_idx)
        embedding.weight.data.requires_grad = True

        if opts['emb_file'] is not None:
            pre_embedding = get_embedding(opts['emb_file'], lang, opts["embedding_key"])
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

        if self.args["encoder_type"] == "birnn":
            self.encoder = BiLSTMEncoder(self.args, self.embedding)
        else:
            raise ValueError("not implemented")
        
        print('Loading gpt model...')
        config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config.pad_token_id = config.eos_token_id
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        

        # config = BartConfig.from_pretrained("facebook/bart-base", output_hidden_states=True, output_attentions=True)
        # self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # config.pad_token_id = config.eos_token_id
        # self.decoder = BartForCausalLM.from_pretrained("facebook/bart-base", config=config)

        # if self.args["decoder_type"] == "pointer_attn":
        #     self.decoder = PointerAttnDecoder(self.args, self.embedding)
        # else:
        #     raise ValueError("not implemented")

        self.reduce_state = ReduceState(self.args)

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.embedding = self.embedding.cuda()
            self.reduce_state = self.reduce_state.cuda()

        self.decoder = torch.nn.DataParallel(self.decoder, device_ids=[0,1])
        print(f'gpt model on {self.decoder.module.device}')

    def get_encode_states(self, batch):
        # NEVER USED
        input_batch = batch['input_batch']
        input_lengths = batch['input_lengths']

        batch_size = input_batch.size(1)
        input_emb = self.embedding(input_batch)
        encoder_outputs, encoder_hidden = self.encoder(input_emb, input_lengths)

        return encoder_outputs, encoder_hidden

    def get_input_from_batch(self, batch):

        enc_batch = batch["input_batch"].transpose(0,1)
        enc_lens = batch["input_lengths"]

        batch_size, max_enc_len = enc_batch.size()
        assert enc_lens.size(0) == batch_size

        enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

        extra_zeros = None
        enc_batch_extend_vocab = None

        if self.args["pointer_gen"]:
            enc_batch_extend_vocab = batch["input_ext_vocab_batch"].transpose(0,1)
            # max_art_oovs is the max over all the article oov list in the batch
            if batch["max_art_oovs"] > 0:
                extra_zeros = Variable(torch.zeros((batch_size, batch["max_art_oovs"])))

        c_t_1 = Variable(torch.zeros((batch_size, 2 * self.args["hidden_size"])))

        coverage = None
        if self.args["is_coverage"]:
            coverage = Variable(torch.zeros(enc_batch.size()))

        if USE_CUDA:
            if enc_batch_extend_vocab is not None:
                    enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
            if extra_zeros is not None:
                extra_zeros = extra_zeros.cuda()
            c_t_1 = c_t_1.cuda()

            if coverage is not None:
                coverage = coverage.cuda()

        return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

    def get_output_from_batch(self, batch):

        dec_batch = batch["target_batch"].transpose(0,1)
        target_batch = batch["target_ext_vocab_batch"].transpose(0,1)
        dec_lens_var = batch["target_lengths"]
        max_dec_len = max(dec_lens_var)

        assert max_dec_len == target_batch.size(1)

        dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

        return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

    def decoded_batch_to_txt(self, all_targets):

        batch_size = all_targets[0].size(0)
        hyp = []
        for i in range(batch_size):
            hyp.append([self.tokenizer.decode(t[i]) for t in all_targets])
        return hyp

    def compute_rouge_reward(self, sent, input_txt, target_txt):
        # not used
        prediction = " ".join("".join(sent))
        if len(prediction.split()) < 1:
            return 0.0
        article = " ".join("".join(input_txt.split()))
        ground_truth = " ".join("".join(target_txt.split()))
        rouge_score = rouge([prediction], [article])[rouge_metric] + self.args["eps"]

    def get_sensation_reward(self, decoded_sents, batch, sensation_model):
        '''
        Gets R_sen
        '''
        #print(f'decoded: {decoded_sents}')
        # print('Is everything in english?: ', np.bitwise_and.reduce(np.array([isEnglish(' '.join(sent)) for sent in decoded_sents])))
        # if not np.bitwise_and.reduce(np.array([isEnglish(' '.join(sent)) for sent in decoded_sents])):
        #     for i, sent in enumerate(decoded_sents):
        #         new_sent = [word if isEnglish(word) else '' for word in sent]
        #         decoded_sents[i] = new_sent
                    
            # print([(isEnglish(' '.join(sent)), sent) for sent in decoded_sents])
            # print('Is everything in english?: ', np.bitwise_and.reduce(np.array([isEnglish(' '.join(sent)) for sent in decoded_sents])))
            # print('==================')
        '''
        # Concatenate the two arguments with a separator tag
        new_batch = input_txt_to_batch(decoded_sents, self.lang)
        separator_sent = ['[SEP]'] * new_batch.shape[0]
        separator = [self.lang.word2idx[w] if w in self.lang.word2idx else UNK_idx for w in separator_sent]
        separator = Variable(torch.LongTensor(separator)).unsqueeze(1)
        if USE_CUDA:
            separator = separator.cuda()
        # print(f"Shape of arguments: {new_batch.shape}, {separator.shape}, {batch['target_batch'].t().shape}")
        new_batch = torch.cat((new_batch, separator, batch['target_batch'].t()), 1)

        # Ensure that the max length of the tensor is 512. Fill with padding.
        batch_size, max_len = new_batch.shape
        padded_batch = torch.full((batch_size, 512), self.lang.word2idx['[PAD]']).type(torch.LongTensor)
        padded_batch[:, :max_len] = new_batch
        padded_batch = padded_batch.to('cuda:1')
        # Prevents the libcublas error?
        '''

        joined_decoded_sents = [' '.join(sent) for sent in decoded_sents]
        sents = [pred + ' [SEP] ' + target for pred, target in zip(joined_decoded_sents, batch['target_txt'])]
        batch = self.bert_tokenizer(sents, return_tensors='pt', padding=True)['input_ids']
        
        # min_out, min_inds = torch.min(batch, dim=1)
        # max_out, max_inds = torch.max(batch, dim=1)
        # print(f'min: {min_out}, max: {max_out}')
        # print(sensation_model)
        if USE_CUDA:
            # batch = batch.to('cuda:1')
            batch = batch.to( next(sensation_model.parameters()).device)

        try:
            with torch.no_grad():
                rewards = sensation_model(batch)
        except RuntimeError:
            print('Runtime Error!')
            print(f'decoded: {decoded_sents}')
            print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
            # print(f'new_batch: {new_batch.shape}, padded_batch: {padded_batch.shape}')
            # print(f'batch: {batch}')
            # print([isEnglish(' '.join(sent)) for sent in decoded_sents])
            
            raise RuntimeError

        # rewards = rewards.cuda()
        rewards = rewards.cuda()
        w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list) if len(word_list) else 1 for word_list in decoded_sents])
        if USE_CUDA:
            w = w.cuda()

        return rewards * w
   
    def get_reward(self, decoded_sents, batch, sensation_model):

        sensation_rewards = self.get_sensation_reward(decoded_sents, batch, sensation_model)
        rewards = sensation_rewards

        rewards = rewards.detach()
        return rewards, 0.0

    def init_batch(self, batch, individual_tokenization=False, max_input_len=None):
        texts = batch['input_txt']
        target_texts = batch['target_txt']

        print(f'Texts: {texts}\nTarget: {target_texts}')

        inputs, targets, batch_size = None, None, 0
        if individual_tokenization:
            inputs = [self.tokenizer(text, return_tensors="pt") for text in texts]
            if max_input_len:
                inputs = [{key: item[:max_input_len] for key, item in inp.items()} for inp in inputs]

            targets = [self.tokenizer(target, return_tensors="pt") for target in target_texts]

            if USE_CUDA:
                # inputs = [{key: item.cuda() for key, item in inp.items()} for inp in inputs]
                # inputs = [{key: item.cuda() for key, item in target.items()} for target in targets]
                print('WARNING: Inputs not placed on GPU')

            batch_size = len(inputs)
        else:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

            self.tokenizer.padding_side = "right"
            targets = self.tokenizer(target_texts, return_tensors="pt", padding=True)

            if USE_CUDA:
                # inputs = {key: item.cuda() for key, item in inputs.items()}
                # targets = {key: item.cuda() for key, item in targets.items()}
                print('WARNING: Inputs not placed on GPU')

            batch_size = inputs['input_ids'].shape[0]

        # padding should stay on the right for targets
        # this ensures it doesn't interfere with teacher forcing

        return inputs, targets, batch_size

    def run_decoder(self, inputs):
        # print(inputs)
        attention_mask = inputs['attention_mask']
        # outputs = self.decoder.generate(**inputs, num_beams=5, max_length=inputs['input_ids'].shape[-1] + 1, return_dict_in_generate=True, output_scores=True)
        # print('calling outputs')
        # total_memory = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             # print(type(obj), obj.size())

        #             acc = 4
        #             for dim in obj.size():
        #                 acc *= dim

        #             total_memory += acc

        #     except:
        #         pass
        # print(f'Storage requirements: {total_memory}')
        print('chkpt1\n', torch.cuda.memory_summary(), '\n')
        outputs = self.decoder(**inputs)
        print('chkpt2\n', torch.cuda.memory_summary(), '\n')

        # get next token
        final_dist = outputs.logits[:, -1, :]
        logits_processor = LogitsProcessorList()
        next_tokens_scores = logits_processor(inputs['input_ids'], final_dist)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        
        # appending next tokens to input_ids
        input_ids = torch.cat((inputs['input_ids'], next_tokens.unsqueeze(0).t()), dim=1)

        # checks if last generated token is EOS, and if so, accordingly updates attention mask
        generated_attention_mask = torch.logical_not((inputs['input_ids'][:, -1].clone().detach() == self.tokenizer.eos_token_id)).long().view(-1, 1)
        attention_mask = torch.cat((attention_mask, generated_attention_mask), dim=1)
        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # del generated_attention_mask
        # del logits_processor
        # del next_tokens_scores
        # del next_tokens
        # gc.collect()
        # print(torch.cuda.memory_summary())

        return inputs, outputs, final_dist
 
    def sample_batch(self, batch):
        # NOT CALLED IN OUR IMPLEMENTATION

        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = self.get_input_from_batch(batch)
        
        # remove this
        # encoder_outputs, encoder_hidden = self.encoder(enc_batch, enc_lens)
        # s_t_1 = self.reduce_state(encoder_hidden)

        # batch_size = enc_batch.size(0)

        inputs, _, batch_size = self.init_batch(batch)

        step_losses = []

        # y_t_1 = Variable(torch.LongTensor([SOS_idx] * batch_size))
        step_mask = Variable(torch.ones(batch_size)).float()
        all_step_mask = []
        if USE_CUDA:
            # y_t_1 = y_t_1.cuda()
            step_mask = step_mask.cuda()
        all_ext_vocab_targets = []
        for di in range(self.args["max_r"]):
            # final_dist, s_t_1,  c_t_1, attn_dist, p_gen, coverage, output1 = self.decoder(y_t_1, s_t_1,
            #                                             encoder_outputs, enc_padding_mask, c_t_1,
            #                                             extra_zeros, enc_batch_extend_vocab,
            #                                                         coverage, di, training=True)
            inputs, _, final_dist = self.run_decoder(inputs)
            
            final_dist = final_dist.softmax(dim=1)
            ext_vocab_target = torch.multinomial(final_dist.data, 1).long().squeeze() # sampling
            all_ext_vocab_targets.append(ext_vocab_target.detach())

            all_step_mask.append(step_mask)
            step_mask = torch.clamp(step_mask - (ext_vocab_target == EOS_idx).float(), min=0.0)

            # y_t_1 = Variable(torch.ones(batch_size)).long()  # sampling
            # for i in range(batch_size):
            #     if ext_vocab_target[i] >= self.vocab_size:
            #         y_t_1[i] = UNK_idx
            #     else:
            #         y_t_1[i] = ext_vocab_target[i]
            # if USE_CUDA:
            #     y_t_1 = y_t_1.cuda()
        ## pad ext_vocab_batch with pad_idx 
        seq_len = self.args["max_r"]
        y_t = Variable(torch.ones(batch_size, seq_len) * PAD_idx).long()
        for i in range(batch_size):
            for j in range(seq_len):
                if all_ext_vocab_targets[j][i] == EOS_idx:
                    y_t[i, j] = EOS_idx
                    break
                y_t[i, j] = all_ext_vocab_targets[j][i]
        if USE_CUDA:
            y_t = y_t.cuda()
            
        all_step_mask = torch.stack(all_step_mask, dim=1)
        target_lens = torch.sum(all_step_mask,dim=1)
        all_ext_vocab_targets = y_t

        ## oov words to unk_token
        all_targets = Variable(torch.ones(batch_size, seq_len) * PAD_idx).long()
        for i in range(batch_size):
            for j in range(seq_len):
                if all_ext_vocab_targets[i, j] >= self.vocab_size:
                    all_targets[i, j] = UNK_idx
                else:
                    all_targets[i, j] = all_ext_vocab_targets[i,j]
        if USE_CUDA:
            all_targets = all_targets.cuda()
        return all_targets, all_ext_vocab_targets, target_lens.long()

    def get_losses(self, batch, sensation_model, use_s_score):
        print('chkpt0\n', torch.cuda.memory_summary(), '\n')
        total_reward, rl_loss, _, rewards_loss, probs = self.get_rl_loss(batch, sensation_model, use_s_score=use_s_score)
        torch.cuda.empty_cache()
        print('chkpt3\n', torch.cuda.memory_summary(), '\n')
        _, ml_loss, _ = self.get_loss(batch, use_s_score=use_s_score)

        # if use_s_score:
        #     loss = rl_loss +  ml_loss
        # else:
        #     loss = (1 - self.args["ml_wt"]) * rl_loss + self.args["ml_wt"] * ml_loss

        return total_reward, rl_loss, ml_loss, rewards_loss, probs

    def get_rl_loss(self, batch, sensation_model, use_s_score):

        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = self.get_input_from_batch(batch)

        # encoder_outputs, encoder_hidden = self.encoder(enc_batch, enc_lens)
        # s_t_1 = self.reduce_state(encoder_hidden)

        # batch_size = enc_batch.size(0)
        inputs, _, batch_size = self.init_batch(batch)
        step_losses = []

        # y_t_1 = Variable(torch.LongTensor([SOS_idx] * batch_size))
        step_mask = Variable(torch.ones(batch_size)).float()
        all_step_mask = []
        if USE_CUDA:
            # y_t_1 = y_t_1.cuda()
            step_mask = step_mask.cuda()
        all_targets = []
        all_output1 = []
        
        # in len of maximum size of headlines
        for di in range(self.args["max_r"]):
            # final_dist, s_t_1,  c_t_1, attn_dist, p_gen, coverage, output1 = self.decoder(y_t_1, s_t_1,
            #                                             encoder_outputs, enc_padding_mask, c_t_1,
            #                                             extra_zeros, enc_batch_extend_vocab,
            #                                                         coverage, di, training=True)

            print(f'get_rl_loss loop {di}')
            inputs, outputs, final_dist = self.run_decoder(inputs)
            # do this to avoid negatives being fed into multinomial
            final_dist = final_dist.softmax(dim=1)
            # print('targets: ', torch.multinomial(final_dist.data, 1).long().squeeze())
            target = torch.multinomial(final_dist.data, 1).long().squeeze() # sampling
            all_targets.append(target.detach())
            # TODO: how to get output1, just use hidden states? that is shape (batch, seq_len, hidden_dim)
            output1 = outputs['hidden_states'][-1][:, -1, :]
            output1 = output1.float()
            # print(f'Output type: {output1.dtype}')
            # this is some hidden state (batch * hidden_dim) -> o_t
            all_output1.append(output1)
            # TODO:why are the gold probs coming from the decoder? seems to be misnamed
            # gold_probs = final_dist[:, target]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.args["eps"])
            
            step_loss = step_loss * step_mask
            all_step_mask.append(step_mask)
            step_losses.append(step_loss)
            step_mask = torch.clamp(step_mask - (target == EOS_idx).float(), min=0.0)
            # print('chkpt0\n', torch.cuda.memory_summary(), '\n')

            # print(all_step_mask[-1].size(), all_targets[-1].size(), all_output1[-1].size(), step_losses[-1].size())
            # y_t_1 = Variable(torch.zeros(batch_size)).long()  # sampling
            # for i in range(batch_size):
            #     if target[i] >= self.vocab_size:
            #         y_t_1[i] = UNK_idx
            #     else:
            #         y_t_1[i] = target[i]
            # if USE_CUDA:
            #     y_t_1 = y_t_1.cuda()

        # this is the linear layer that calculates \hat{R}_t
        # print(all_output1, all_step_mask)
        print('chkpt1\n', torch.cuda.memory_summary(), '\n')
        baseline_rewards = [self.expected_reward_layer(output1.detach()) * step_mask.unsqueeze(1).detach() \
                                            for output1, step_mask in zip(all_output1, all_step_mask)]
        # print('chkpt2\n', torch.cuda.memory_summary(), '\n')
        baseline_rewards = torch.cat(baseline_rewards, dim=1)
        all_step_mask = torch.stack(all_step_mask, dim=1).float()
        dec_lens_var = torch.sum(all_step_mask,dim=1)
        # print('chkpt3\n', torch.cuda.memory_summary(), '\n')
        decoded_sents = self.decoded_batch_to_txt(all_targets)
        # print(decoded_sents, len(decoded_sents))
        total_reward, probs = self.get_reward(decoded_sents, batch, sensation_model)
        total_reward = total_reward.unsqueeze(1)
        
        # getting (R - \hat{R}_t)
        # torch.Size([batch, 1]) torch.Size([batch, max_r])
        # print(total_reward.shape, baseline_rewards.shape)
        reward =  total_reward.detach() - baseline_rewards.detach()
        # print('chkpt4\n', torch.cuda.memory_summary(), '\n')
        
        sum_losses = torch.sum(reward * torch.stack(step_losses, 1), 1)
        # this is for ARL
        if use_s_score:
            batch_avg_loss = sum_losses/dec_lens_var.float()*(1 - batch["sensation_scores"])
        else:
            batch_avg_loss = sum_losses/dec_lens_var.float()
        rl_loss = torch.mean(batch_avg_loss)
        # _, ml_loss, _ = self.get_loss(batch, use_s_score=use_s_score)
        # if use_s_score:
        #     loss = rl_loss +  ml_loss
        # else:
        #     loss = (1 - self.args["ml_wt"]) * rl_loss + self.args["ml_wt"] * ml_loss

        rewards_loss = torch.sum((total_reward - baseline_rewards) ** 2 * all_step_mask) / torch.sum(all_step_mask)
        print('chkpt2\n', torch.cuda.memory_summary(), '\n')
        # return total_reward.mean(), loss, Variable(torch.FloatTensor([0.0])), rewards_loss, probs
        return total_reward.mean(), rl_loss, Variable(torch.FloatTensor([0.0])), rewards_loss, probs

    def get_loss(self, batch, use_s_score=False, return_full_loss=False):
        # calculates MLE loss
        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = self.get_input_from_batch(batch)
        # seems like target and dec batches are the same
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = self.get_output_from_batch(batch)

        # encoder_outputs, encoder_hidden = self.encoder(enc_batch, enc_lens)
        # s_t_1 = self.reduce_state(encoder_hidden)

        # batch_size = enc_batch.size(0)
        inputs, targets, batch_size = self.init_batch(batch)
        step_losses = []

        # y_t_1 = Variable(torch.LongTensor([SOS_idx] * batch_size))
        # if USE_CUDA:
        #     y_t_1 = y_t_1.cuda()

        for di in range(min(targets['input_ids'].shape[-1], self.args["max_r"])):
            # final_dist, s_t_1,  c_t_1, attn_dist, p_gen, coverage, _ = self.decoder(y_t_1, s_t_1,
            #                                             encoder_outputs, enc_padding_mask, c_t_1,
            #                                             extra_zeros, enc_batch_extend_vocab,
            #                                                         coverage, di, training=True)
            print(f'get_loss loop {di}')
            print(f'chkpt4.{di}\n {torch.cuda.memory_summary()}\n')
            inputs, _, final_dist = self.run_decoder(inputs)
            # final_dist = Variable(final_dist, requires_grad=True).cuda()
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.args["eps"])
            if self.args["is_coverage"]:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + self.args["cov_loss_wt"] * step_coverage_loss
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
            # TODO: how to do teacher forcing here? I could try setting hidden state
            #       but once again, the shapes differ
            # TODO: forces to be teacher's answer and calculates loss based students? what is the teacher and why is this not performing
            # y_t_1 = dec_batch[:, di]  # Teacher forcing
            # print(inputs['input_ids'].shape, targets['input_ids'].shape, di)
    
            inputs['input_ids'][:, -1] = targets['input_ids'][:, di] # Teacher forcing

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        if use_s_score:
            batch_avg_loss = sum_losses/dec_lens_var.float()*batch["sensation_scores"]
        else:
            batch_avg_loss = sum_losses/dec_lens_var.float()
        loss = torch.mean(batch_avg_loss)

        if return_full_loss:
            return None, loss, Variable(torch.FloatTensor([0.0])), batch_avg_loss
        else:
            return None, loss, Variable(torch.FloatTensor([0.0]))

    def get_prob(self, batch):

        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = self.get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = self.get_output_from_batch(batch)

        # encoder_outputs, encoder_hidden = self.encoder(enc_batch, enc_lens)
        # s_t_1 = self.reduce_state(encoder_hidden)

        # batch_size = enc_batch.size(0)
        inputs, targets, batch_size = self.init_batch(batch)
        step_losses = []

        # y_t_1 = Variable(torch.LongTensor([SOS_idx] * batch_size))
        # if USE_CUDA:
        #     y_t_1 = y_t_1.cuda()

        for di in range(min(targets['input_ids'].shape[-1], self.args["max_r"])):
            # final_dist, s_t_1,  c_t_1, attn_dist, p_gen, coverage, _ = self.decoder(y_t_1, s_t_1,
            #                                             encoder_outputs, enc_padding_mask, c_t_1,
            #                                             extra_zeros, enc_batch_extend_vocab,
            #                                                         coverage, di, training=True)
            inputs, _, final_dist = self.run_decoder(inputs)

            target = target_batch[:, di]

            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.args["eps"])
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
            # y_t_1 = dec_batch[:, di]  # Teacher forcing
            # print(inputs['input_ids'].shape, targets['input_ids'].shape, di)
            inputs['input_ids'][:, -1] = targets['input_ids'][:, di] # Teacher forcing

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var.float()
        loss = torch.mean(batch_avg_loss)

        return loss

    def format_output(self, decoder_outputs, batch_size):

        ## move all decoder_outputs after EOS as pad
        sampled_length = Variable(torch.zeros(batch_size)).long()
        sampled_outputs = Variable(torch.ones(batch_size,self.max_r) * PAD_idx).long()
        if USE_CUDA:
            sampled_outputs = sampled_outputs.cuda()
            sampled_length = sampled_length.cuda()
        for i in range(batch_size):
            for j in range(self.max_r):
                sampled_outputs[i,j] = decoder_outputs[j][i]
                if decoder_outputs[j][i] == EOS_idx:
                    sampled_length[i] = j+1 
                    break

        return sampled_outputs, sampled_length

    def get_log_prob_of_sample(self, sample, decoder_hidden, encoder_outputs):
        ## get log prob of sample
        # NOT USED
        sample = sample.transpose(0,1) ## seq_len * batch_size
        batch_size = sample.size(1)
        decoder_inputs = Variable(torch.LongTensor([SOS_idx] * batch_size))
        if USE_CUDA:
            decoder_inputs = decoder_inputs.cuda()
        log_probs = []
        for t in range(self.max_r):
            decoder_emb = self.embedding(decoder_inputs)
            if USE_CUDA:
                decoder_emb = decoder_emb.cuda()
            decoder_vocab, decoder_hidden = self.decoder(decoder_emb, decoder_hidden, encoder_outputs)
            decoder_vocab = F.log_softmax(decoder_vocab, 1)
            decoder_inputs = sample[t]
            log_probs.append(torch.gather(decoder_vocab, 1, sample[t].view(-1,1))) ## batch_size * 1

        return torch.cat(log_probs, 1) # batch_size * max_r

    def batch_to_txt(self, input_batch, input_length):

        decoded_sents = []
        for i, input_i in enumerate(input_batch):
            if USE_CUDA:
                decoded_sents.append([self.lang.idx2word[int(ni.cpu().numpy())] for ni in input_i[:int(input_length[i])]])
            else:
                decoded_sents.append([self.lang.idx2word[int(ni.numpy())] for ni in input_i[:int(input_length[i])]])
        return decoded_sents

    def decode_batch(self, batch, decode_type):

        self.encoder.train(False)
        self.decoder.module.train(False)

        assert decode_type == "beam"
        inputs, _, _ = self.init_batch(batch, individual_tokenization=True)
        if USE_CUDA:
            inputs = [{key: item.cuda() for key, item in inp.items()} for inp in inputs]

        outputs = [self.decoder.module.generate(**inp, num_beams=5, max_length=256) for inp in inputs]
        # print(outputs[0], type(outputs[0]))
        decoded_sents = [self.tokenizer.decode(output[0]) for output in outputs]
        # beam_sh = BeamSearch(self, self.args, self.lang)
        # decoded_sents = beam_sh.beam_search(batch)

        self.encoder.train(True)
        self.decoder.module.train(True)

        return decoded_sents

    def get_rep_rate(self, sents):
        num_uni_tokens, num_tokens = 0, 0
        for sent in sents:
            tokens = sent.strip().split()
            num_uni_tokens += len(set(tokens))
            num_tokens += len(tokens)
        return 1.  - num_uni_tokens * 1.0 / num_tokens
        
    def evaluate(self, dev, decode_type, return_pred=False, sensation_model=None):

        logging.info("start evaluation")
        hyp = []
        ref = []
        tmp_loss = []
        rewards = []
        sensation_scores = []
        articles = []

        for j, data_dev in enumerate(dev):
            l = self.get_prob(data_dev)
            tmp_loss.append(float(l.data.cpu().numpy()))

            decoded_sents = self.decode_batch(data_dev, decode_type)
            for i, sent in enumerate(decoded_sents):
                hyp.append(" ".join(sent)) 
                ref.append(" ".join(data_dev["target_txt"][i].split()))
                articles.append(data_dev["input_txt"][i])
            if sensation_model is not None:
                rewards.extend([r for r in self.get_reward(decoded_sents, data_dev, sensation_model)[0]])
            if "sensation_scores" in data_dev:
                sensation_scores.extend([float(s) for s in data_dev["sensation_scores"]])

        rouge_score = rouge(hyp, ref)
        logging.info("decode type: {}, score: {}, ref repeatition rate: {}, prediction repeatition rate: {}".format(decode_type, rouge_score, self.get_rep_rate(ref), self.get_rep_rate(hyp)))
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
            # hyp.append("  ".join("".join(sent)))
            hyp.append(" ".join(sent))
            # ref.append(" ".join("".join(batch["target_txt"][i].split())))
            ref.append(batch["target_txt"][i])
        return hyp, ref
                
