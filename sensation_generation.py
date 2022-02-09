import os
import numpy as np
import logging
from tqdm import tqdm
from utils.config import *
from utils.data_utils import *
from torch.nn.utils import clip_grad_norm
from models.batch_utils import *
from models.sensation_scorer import PersuasivenessClassifier
import logging
import copy
from utils.rouge import rouge
from utils.function import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer(object):
    def __init__(self):

        args = NNParams().args

        with open("vocab.txt", "r") as f:
            vocab = f.readlines()
        vocab = list(map(lambda x: x.strip().replace('#',''), vocab))


        train, dev, test, max_q, max_r = prepare_data_seq(batch_size=args['batch_size'], shuffle=True, thd=args["thd"])
        args["max_q"] = max_q
        args["max_r"] = max_r

        self.args = args
        print(args)
        self.train = train
        self.dev = dev
        self.test = test

        model = PointerAttnSeqToSeq(self.args)
        self.model = model
        if USE_CUDA:
            self.model = self.model.to("cuda:0")
            print('Gen model is on: ', next(self.model.parameters()).device)

        logging.info(model)
        logging.info("encoder parameters: {}".format(count_parameters(model.encoder)))
        logging.info("decoder parameters: {}".format(count_parameters(model.decoder)))
        logging.info("embedding parameters: {}".format(count_parameters(model.embedding)))
        logging.info("model parameters: {}".format(count_parameters(model)))

        self.loss, self.acc, self.reward, self.print_every = 0.0, 0.0, 0.0, 1

        print('Loading sensation model...')
        self.sensation_model = PersuasivenessClassifier(self.lang)
        self.sensation_model.load_state_dict(torch.load("persuasive_model.pt"))
        self.sensation_model.bert.resize_token_embeddings(len(vocab))
        if USE_CUDA:
            self.sensation_model.to('cuda:1')
            print('Sensation model is on: ', next(self.sensation_model.parameters()).device)

        if self.args['optimizer'] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        elif self.args['optimizer'] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'])
        else:
            raise ValueError("optimizer not implemented")

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
            r, loss, expected_rewards_loss = self.model.get_rl_loss(batch, self.sensation_model, use_s_score=self.args["use_s_score"])
        else:
            loss = self.model.get_loss(batch)

        # loss = Variable(loss, requires_grad = True)
        # loss.backward(create_graph = True)
        loss.backward()

        clip_grad_norm(self.model.parameters(), self.args["max_grad_norm"])

        self.optimizer.step()

        self.loss += loss.data 
        if self.args["use_rl"]:
            self.reward += r.data
            
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
                self.model.expected_reward_layer = self.model.expected_reward_layer.to("cuda:0")
            self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            step, best_metric = self.load_rl_model()
        elif self.args["path"] is not None:
            # step, best_metric = self.load_base_model() 
            if self.args["use_rl"]:
                best_metric = 0.0
                self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
                if USE_CUDA:
                    self.model.expected_reward_layer = self.model.expected_reward_layer.to("cuda:0")
                self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
        else:
            pass
        self.old_model = copy.deepcopy(self.model) 
        total_steps = self.args["total_steps"]
        while step < total_steps:
            for j, batch in enumerate(self.train):

                if self.args['debug'] and j>1100:
                    break
                
                if not self.args["debug"]:
                    logging_step = 1000
                else:
                    logging_step = 10

                if j % logging_step == 0 and j:

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

if __name__ == "__main__":
    trainer =  Trainer()
    trainer.training()
