import numpy as np
import logging
from tqdm import tqdm
from sutils.config import *
from sutils.utils_sensation_lcsts import *
from torch.nn.utils import clip_grad_norm
from seq2seq.sensation_get_to_the_point import *
from seq2seq.sensation_scorer import SensationCNN, PersuasivenessClassifier
import logging
import copy
import jieba
from sutils.function import *
import sys

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig


# from persuasiveness_classifier import PersuasivenessClassifier, get_persuasive_pairs_xml
from transformers import BertTokenizer

class PersuasivePairsDataset(Dataset):
    """Persuasive pairs dataset."""

    def __init__(
        self,
        persuasive_pairs_df,
        columns: dict = {'label': 'label',
                         'sentence_a': 'sentence_a', 'sentence_b': 'sentence_b'},
    ):
        """
        Args:
            persuasive_pairs_df (pd.DataFrame): dataframe of persuasive pairs
            columns (dict): columns with default column names paired with custom ones
        """
        self.persuasive_pairs_df = persuasive_pairs_df
        self.columns = columns

    def __len__(self):
        return len(self.persuasive_pairs_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.data)

        data_idx = self.persuasive_pairs_df[idx]
        label = data_idx[self.columns['label']]
        sentence_a = data_idx[self.columns['sentence_a']]
        sentence_b = data_idx[self.columns['sentence_b']]

        return {
            'texts': [sentence_a, sentence_b],
            'label': int(label)
        }

def get_data_loaders(args, tokenizer,
                     train_dir='../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/',
                     test_dir='../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/'):
    '''
    Gets the train and test dataloaders from the data loader function.

    Args:
        args: command-line arguments
        tokenizer: tokenizer for tokenizing samples
        dataset_type: type of dataset corresponding to model type
        data_loader_fn: function to load data from directories
        train_dir: directory to load training data from
        test_dir: directory to load testing data from
    Returns:
        a train and test dataloader
    '''

    train_dataset, test_dataset = None, None
    train_dataloader, test_dataloader = None, None

    if train_dir == test_dir:
        # data is in the same file/directory
        persuasive_pairs = data_loader_fn(
            train_dir,
            exclude_equal_arguments=args.exclude_equal_arguments
        )

        # random train/test split
        total_length = len(persuasive_pairs)
        random.shuffle(persuasive_pairs)
        split = int(args.train_test_split * total_length)
        train_dataset = persuasive_pairs[:split]
        test_dataset = persuasive_pairs[split:]
    else:
        train_dataset = data_loader_fn(
            train_dir,
            exclude_equal_arguments=args.exclude_equal_arguments
        )
        test_dataset = data_loader_fn(
            test_dir,
            exclude_equal_arguments=args.exclude_equal_arguments
        )

    if dataset_type == PersuasivePairsDataset:
        persuasive_pairs_train = dataset_type(train_dataset, tokenizer)
        persuasive_pairs_test = dataset_type(test_dataset, tokenizer)
    else:
        persuasive_pairs_train = dataset_type(train_dataset)
        persuasive_pairs_test = dataset_type(test_dataset)

    if args.do_train:
        train_dataloader = DataLoader(persuasive_pairs_train, batch_size=args.train_batch_size,
                                      shuffle=True, num_workers=0)
    if args.do_eval:
        test_dataloader = DataLoader(persuasive_pairs_test, batch_size=args.test_batch_size,
                                     shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer(object):
    def __init__(self):

        args = NNParams().args
        # args['batch_size'] = 4
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # lang = Lang(list(tokenizer.encoder.keys()))
        with open("vocab.txt", "r") as f:
            vocab = f.readlines()
        vocab = list(map(lambda x: x.strip().replace('#',''), vocab))
        # print(vocab)
        # setting args for our data loader function
        # args.exclude_equal_arguments = True
        # args.train_test_split = 0.8
        # args.do_train = True
        # args.do_eval = True
        # train, dev = get_data_loaders(args, tokenizer)
        # test = dev

        train, dev, test, lang, max_q, max_r = prepare_data_seq(batch_size=args['batch_size'], debug=args["debug"], shuffle=True, pointer_gen=args["pointer_gen"], vocab=vocab, thd=args["thd"])
        args["vocab_size"] = lang.n_words
        print(args["vocab_size"])
        args["output_vocab_size"] = lang.n_words
        args["max_q"] = max_q
        args["max_r"] = max_r

        self.args = args
        print(args)
        self.train = train
        self.dev = dev
        self.test = test
        self.lang = lang

        # model = globals()[args["model_type"]](args, lang, max_q, max_r)
        model = PointerAttnSeqToSeq(self.args, lang)
        self.model = model
        if USE_CUDA:
            self.model = self.model.cuda()
            print('Gen model is on: ', next(self.model.parameters()).device)

        logging.info(model)
        logging.info("encoder parameters: {}".format(count_parameters(model.encoder)))
        logging.info("decoder parameters: {}".format(count_parameters(model.decoder)))
        logging.info("embedding parameters: {}".format(count_parameters(model.embedding)))
        logging.info("model parameters: {}".format(count_parameters(model)))

        self.loss, self.acc, self.reward, self.print_every = 0.0, 0.0, 0.0, 1

        # assert args["sensation_scorer_path"] is not None
        # opts = torch.load(args["sensation_scorer_path"]+"/args.th")
        # self.sensation_model = SensationCNN(opts, self.lang)
        # logging.info("load checkpoint from {}".format(args["sensation_scorer_path"]))
        # checkpoint = torch.load(args["sensation_scorer_path"]+"/sensation_scorer.th")
        # self.sensation_model.load_state_dict(checkpoint['model'])
        print('Loading sensation model...')
        self.sensation_model = PersuasivenessClassifier(self.lang)
        # sys.path.insert(0, '../persuasive_classifier/models')
        self.sensation_model.load_state_dict(torch.load("persuasive_model.pt"))
        self.sensation_model.bert.resize_token_embeddings(len(vocab))
        if USE_CUDA:
            # self.sensation_model.to('cuda:1')
            self.sensation_model.cuda()
            print('Sensation model is on: ', next(self.sensation_model.parameters()).device)

        if self.args['optimizer'] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        elif self.args['optimizer'] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'])
        else:
            raise ValueError("optimizer not implemented")

        ds_config = './ds_config.json'
        dschf = HfDeepSpeedConfig(ds_config)
        self.engine = deepspeed.initialize(model=self.model, config_params=ds_config, optimizer=self.optimizer)

    def save_model(self, save_name, best_result, step):
        directory = "sensation_save/" + save_name + "/"
        directory = directory + "_".join([str(self.args[a]) for a in save_params]) + "_" + str(best_result)
        if not os.path.exists(directory):
            os.makedirs(directory)
        ckpt = {"model": self.model.state_dict(),  "step": step, "optimizer": self.optimizer.state_dict(),
         "best_result":best_result}
        torch.save(self.args, directory+"/args.th")
        if self.args["use_rl"]:
            ckpt["rl_optimizer"] = self.rl_optimizer
            torch.save(ckpt, directory+"/rl.th")
        else:
            torch.save(ckpt, directory+"/get_to_the_point.th")
        return directory
 
    def load_base_model(self):
        path = self.args["path"]
        ckpt = torch.load(path+"/get_to_the_point.th")
        logging.info("load ckpt from {}, step is {}, best_result {}".format(path, ckpt["step"], ckpt["best_result"]))

        self.model.load_state_dict(ckpt["model"])
        if not self.args["use_rl"]:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["step"], ckpt["best_result"]

    def load_rl_model(self):
        path = self.args["rl_model_path"]
        ckpt = torch.load(path+"/rl.th")
        logging.info("load ckpt from {}, step is {}, best_result {}".format(path, ckpt["step"], ckpt["best_result"]))

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.rl_optimizer = ckpt["rl_optimizer"]
        return ckpt["step"], ckpt["best_result"]

    def print_loss(self, step):
        print_loss_avg = self.loss / self.print_every
        print_acc_avg = self.acc / self.print_every
        print_reward_avg = self.reward / self.print_every
        if self.args["use_rl"]:
            print_expected_rewards_loss_avg = self.expected_rewards_loss / self.print_every
        self.print_every += 1
        if self.args["use_rl"]:
            return f'step: {step}, L:{print_loss_avg}, acc:{print_acc_avg}, r:{print_reward_avg}, r_loss:{print_expected_rewards_loss_avg}'
        else:
            return f'step: {step}, L:{print_loss_avg}, acc:{print_acc_avg}, r:{print_reward_avg}'


    def train_step(self, batch, step, reset):

        if reset:
            self.loss = 0.0
            self.acc = 0.0
            self.reward = 0.0
            self.print_every = 1
            if self.args["use_rl"]:
                self.expected_rewards_loss = 0.0

        self.optimizer.zero_grad()
        assert self.args["use_s_score"] is not None
        if self.args["use_rl"]:
            r, loss, acc, expected_rewards_loss, _ = self.model.get_rl_loss(batch, self.sensation_model, use_s_score=self.args["use_s_score"])
        else:
            _, loss, acc = self.model.get_loss(batch)

        # loss = Variable(loss, requires_grad = True)
        # loss.backward(create_graph = True)
        loss.backward()

        clip_grad_norm(self.model.parameters(), self.args["max_grad_norm"])

        self.optimizer.step()

        self.loss += loss.data 
        self.acc += acc.data
        if self.args["use_rl"]:
            self.reward += r.data

        if self.args["use_rl"]:
            self.rl_optimizer.zero_grad()
            expected_rewards_loss.backward()
            self.rl_optimizer.step()
            self.expected_rewards_loss += expected_rewards_loss.data

    def training(self):
        # Configure models
        step = 0
        best_metric = 0.0
        cnt = 0

        if self.args["use_rl"] and self.args["path"] is None and self.args["rl_model_path"] is None:
            raise ValueError("use rl but path is not given")

        if self.args["use_rl"] is None and self.args["rl_model_path"] is not None:
            raise ValueError("not using rl but give rl_model_path")

        if self.args["rl_model_path"] is not None:
            self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
            if USE_CUDA:
                self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
            self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            step, best_metric = self.load_rl_model()
        elif self.args["path"] is not None:
            # step, best_metric = self.load_base_model() 
            if self.args["use_rl"]:
                best_metric = 0.0
                self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
                if USE_CUDA:
                    self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
                self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
        else:
            pass
        self.old_model = copy.deepcopy(self.model) 
        total_steps = self.args["total_steps"]
        while step < total_steps:
            for j, batch in enumerate(self.train):
                # print(batch)
                #print('Decoding: ')
                #====
                # decoded_sents = self.model.decode_batch(batch,"beam")
                #print('Decoded:', decoded_sents)
                #print(len(decoded_sents), [len(sent) for sent in decoded_sents])
                #====
                # return
                if self.args['debug'] and j>1100:
                    break
                
                if not self.args["debug"]:
                    logging_step = 1000
                else:
                    logging_step = 10

                if j % logging_step == 0 and j:
                    # if self.args["use_rl"]:
                    #     save_folder = "logs/Rl/"+"_".join([str(self.args[a]) for a in save_params]) 
                    #     os.makedirs(save_folder, exist_ok=True)
                    #     self.save_decode_sents(self.test, save_folder+"/prediction_step_{}.txt".format(step))

                    hyp, ref = self.model.predict_batch(batch, self.args["decode_type"])
                    old_hyp, _ = self.old_model.predict_batch(batch, self.args["decode_type"])
                    decoded_sents = self.model.decode_batch(batch,"beam")
                    print(f'Original: {batch["input_txt"]}\n Decoded: {decoded_sents}')
                    sensation_rewards = self.model.get_sensation_reward(decoded_sents, batch, self.sensation_model)
                    rewards = self.model.get_reward(decoded_sents, batch, self.sensation_model)[0]
                    for i,(prediction, ground_truth, old_pred) in enumerate(zip(hyp, ref, old_hyp)):
                        logging.info("prediction: {}".format(prediction))
                        logging.info("seq2seq prediction: {}".format(old_pred))
                        logging.info("prediction sensation score: {}, {}".format(sensation_rewards[i], rewards[i]))
                        if self.args["use_rl"]:
                            rouge_rewards = self.model.compute_rouge_reward(list(jieba.cut("".join(prediction.split()))), batch["input_txt"][i], batch["target_txt"][i])
                            logging.info("rouge_r: {}, reward:{}".format(rouge_rewards, rewards[i]))
                        logging.info("ground truth: {}".format(ground_truth))
                        logging.info("ground sensation score: {}".format(batch["sensation_scores"][i]))
                        logging.info("input article: {}".format(batch["input_txt"][i]))
                        logging.info("decode type: {}, {}: {}".format(self.args["decode_type"], rouge_metric, rouge([prediction], [ground_truth])[rouge_metric]))

                if step % int(self.args['eval_step']) == 0 and j: 
                    dev_metric, _, (hyp, ref, rewards, sensation_scores, articles) = self.model.evaluate(self.dev, self.args["decode_type"], sensation_model=self.sensation_model, return_pred=True)
                    if(dev_metric > best_metric):
                        best_metric = dev_metric
                        cnt=0
                        if self.args["use_rl"]:
                            print('Saving model...')
                            directory = self.save_model("Rl", best_metric, step)
                            with open(directory + "/prediction", "w") as f:
                                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                        else:
                            directory = self.save_model("PointerAttn", best_metric, step)
                            with open(directory + "/prediction", "w") as f:
                                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                    else:
                        cnt+=1
                    if(cnt == 5): 
                        ## early stopping
                        step = total_steps + 1
                        break

                self.train_step(batch, step, j==0)
                logging.info(self.print_loss(step))
                step += 1

                torch.cuda.empty_cache()


    def save_decode_sents(self, data, save_file):

        logging.info("start decoding")
        hyp = []
        ref = []
        article = []
        # pbar = tqdm(enumerate(dev), total=len(dev))
        # for j, data_dev in pbar:
        rewards = []
        rouge_r = []
        sensation_rewards = []
        for j, data_dev in enumerate(data):

            decoded_sents = self.model.decode_batch(data_dev, "beam")
            if self.args["use_rl"]:
                sensation_rewards.extend([r for r in self.model.get_sensation_reward(decoded_sents, data_dev, self.sensation_model)])
                rewards.extend([ r for r in self.model.get_reward(decoded_sents, data_dev,
                 self.sensation_model)[0] ])
            for i, sent in enumerate(decoded_sents):
                hyp.append(" ".join("".join(sent)))
                ref.append(" ".join("".join(data_dev["target_txt"][i].split())))
                article.append(data_dev["input_txt"][i])
                if self.args["use_rl"]:
                    rouge_r.append(self.model.compute_rouge_reward(sent, data_dev["input_txt"][i], data_dev["target_txt"][i]))

        rouge_score = rouge(hyp, ref)
        with open(save_file, "w") as f:
            if self.args["use_rl"]:
                f.write("\n".join(["{}\nrouge_r: {},sensation_reward:{}, reward:{}\n{}\n{}\n".format(h,r_r,l_r,r,g,a) for h,g,r_r,l_r,r,a in zip(hyp, ref, rouge_r,sensation_rewards, rewards, article)]))
            else:
                f.write("\n".join([h+"\n"+g+"\n" for h,g in zip(hyp, ref)]))
            f.write("\n" + str(rouge_score) + "\n")
            f.write("rewards: " + str(sum(rewards) / len(rewards)) + "\n")

    def decoding(self, decode_type="beam"):
        # Configure models

        if self.args["use_rl"]  and self.args["rl_model_path"] is None:
            raise ValueError("use rl but path is not given")

        if self.args["use_rl"] is None and self.args["rl_model_path"] is not None:
            raise ValueError("not using rl but give rl_model_path")

        if self.args["rl_model_path"] is not None:
            self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
            if USE_CUDA:
                self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
            self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            step, _ = self.load_rl_model()
            save_file = self.args["rl_model_path"] + "/prediction.txt"
        elif self.args["path"] is not None:
            step, _ = self.load_base_model()
            if self.args["use_rl"]:
                self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
                if USE_CUDA:
                    self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
                self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            save_file = self.args["path"] + "/prediction.txt"
        else:
            pass

        _, _, (hyp, ref, rewards, sensation_scores, articles) = self.model.evaluate(self.test, self.args["decode_type"], sensation_model=self.sensation_model, return_pred=True)
        if self.args["rl_model_path"] is not None:
            directory = self.args["rl_model_path"]
            with open(directory + "/test_prediction", "w") as f:
                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                f.write("\n{}\n".format(str(rouge(hyp, ref))))
                
        elif self.args["path"] is not None:
            directory = self.args["path"]
            with open(directory + "/test_prediction", "w") as f:
                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                f.write("\n{}\n".format(str(rouge(hyp, ref))))

if __name__ == "__main__":
    trainer =  Trainer()
    trainer.training()
