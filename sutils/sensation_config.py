import argparse
import logging
from sutils.global_variables import *

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3

rouge_metric = "rouge_l/f_score"
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))

class CNNParams(object):
    def __init__(self):
        self.set_args()
        # self.set_parameters()


    def set_args(self):
        parser = argparse.ArgumentParser(description="argument for nn parameters")

        parser.add_argument('-rl_cov', type=str, default="rl_no_cov", help="rl coverage")
        parser.add_argument("-rouge_wt", type=float, default=None, help="rouge wt for combining")

        # seq2seq parameters
        ## nn parameters
        parser.add_argument('-batch_size', type=int, default=32, help="batch size")
        parser.add_argument('-emb_size', type=int, default=300, help="embedding size")
        parser.add_argument('-output_vocab_size', type=int, default=50000, help="output_vocab_size")
        parser.add_argument('-hidden_size', type=int, default=768, help="hidden size")
        parser.add_argument('-dropout', type=float, default=0.0, help="dropout rate")
        
        parser.add_argument('-decode_type', type=str, default="beam", help="decoding method for generation")
        parser.add_argument('-encoder_type', type=str, default="birnn", help="encoder_type, rnn, birnn")
        parser.add_argument('-decoder_type', type=str, default="pointer_attn", help="decoder_type, rnn, attn_rnn, pointer_attn")
        parser.add_argument('-encoder_layers', type=int, default=1, help="number of encoder layers")
        parser.add_argument('-decoder_layers', type=int, default=1, help="number of decoder layers")
        # parser.add_argument('-teacher_forcing_rate', type=float, default=1.0, help="teacher_forcing_rate")
        parser.add_argument('-min_dec_steps', type=int, default=1, help="min length for generation")
        parser.add_argument('-max_dec_step', type=int, default=1000, help="max length for decoding")
        # parser.add_argument('-emoji_dim', type=int, default=12, help="dimension for emoji")

        parser.add_argument("-num_multinomial_samples", type=int, default=10, help="number of samples in multinomial sampling")
        parser.add_argument("-num_roll_outs", type=int, default=10, help="number of roll outs for each time step in rl")
        parser.add_argument("-use_oov_emb", type=bool, default=False, help="use oov embedding")


        parser.add_argument("-adv_training", type=bool, default=False, help="use adv training or not")
        parser.add_argument("-use_rl", type=bool, default=False, help="use rl or not")
        parser.add_argument("-rl_ratio", type=float, default=1.0, help="ratio of rl_loss in rl")
        parser.add_argument("-rl_lr", type=float, default=0.001, help="learning rate of rl")
        # discriminator parameter
        # parser.add_argument('-ds', type=float, default=0.0, help="dropout rate")
        # parser.add_argument('-dropout', type=float, default=0.0, help="dropout rate")

        ## cnn discriminator parameters
        parser.add_argument('-num_filters', type=int, default=512, help="number cnn filters")
        parser.add_argument('-filter_sizes', type=str, default="1,3,5", help="filter sizes")

        parser.add_argument('-beam_size', type=int, default=5, help="beam size")

        # adv_training
        parser.add_argument('-d_steps', type=int, default=1, help="steps to train discriminator")
        parser.add_argument('-g_steps', type=int, default=1, help="steps to train generator")
        parser.add_argument('-d_lr', type=float, default=0.001, help="learning  rate for CNN")
        parser.add_argument('-d_model_path', type=str, default=None, help="disc model path")
        parser.add_argument('-combined_method', type=str, default="plus", help="how to combine rouge and adv")
        parser.add_argument('-adv_model_path', type=str, default=None, help="load existing adv model")

        ## optimization
        parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
        parser.add_argument('-decay_lr', type=int, default=3, help="decay learning rate if validation is not improving")
        parser.add_argument('-epochs', type=int, default=100, help="epochs for runing")
        parser.add_argument('-total_steps', type=int, default=100000000, help="total steps for training")
        parser.add_argument('-optimizer', type=str, default="adam", help="which optimizer to use")
        parser.add_argument('-max_grad_norm', type=float, default=2.0, help="max grad norm")

        ## pointer_gen parameters
        parser.add_argument('-pointer_gen', type=bool, default=True, help="use pointer generator or not")
        parser.add_argument('-is_coverage', action='store_true', help="use coverage or not, default as False")
        parser.add_argument('-cov_loss_wt', type=float, default=1.0, help="coverage loss weight")
        parser.add_argument('-eps', type=float, default=1e-12, help="epison to avoid 0 probs")

        ## other args
        parser.add_argument('-debug', type=bool, default=False, help="debug or not")
        # parser.add_argument('-save_dir', type=str, default=None, help="saving path")
        parser.add_argument('-sensation_scorer_path', type=str, default=None, help="load existing sensation model")
        parser.add_argument('-rl_model_path', type=str, default=None, help="load existing rl model")
        parser.add_argument('-path', type=str, default=None, help="load existing path")
        parser.add_argument('-evalp', type=int, default=1, help="evaluation epoch")
        parser.add_argument('-eval_step', type=int, default=10000, help="evaluation steps")
        parser.add_argument('-model_type', type=str, default=None, help="model type")
        parser.add_argument('-emb_file', type=str, default=None)
        parser.add_argument('-use_pretrained_emb', type=bool, default=False)

        self.args = vars(parser.parse_args())
        # if self.args["use_pretrained_emb"]:
        #     self.args['embedding_key'] = "id2word"
        #     self.args['emb_file'] = glove_emb
        #     self.args['emb_file']["vocab_path"] = glove_emb["vocab_path"].format(self.args["emb_size"])
        #     self.args['emb_file']["embedding_path"] = glove_emb["embedding_path"].format(self.args["emb_size"])
        # if self.args['use_emo2vec']:
        # 	self.args['emb_file']["vocab_path"] = self.args['emb_file']["vocab_path"].format(self.args["emb_size"] - 100)
        # 	self.args['emb_file']["embedding_path"] = self.args['emb_file']["embedding_path"].format(self.args["emb_size"] - 100)
        # else:
        # 	self.args['emb_file']["vocab_path"] = self.args['emb_file']["vocab_path"].format(self.args["emb_size"])
        # 	self.args['emb_file']["embedding_path"] = self.args['emb_file']["embedding_path"].format(self.args["emb_size"])
        logging.info(self.args)


    def set_parameters(self):
        if self.args.config_file is not None:
            configs = yaml.load(open(self.config_file, 'r'))


