import nltk
import logging
import numpy as np

from dutils.losses import get_prob
from dutils.batch_utils import decode_batch
from dutils.sensation_scorer import get_discriminator_reward

from dutils.rouge import rouge

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


class Evaluation():
    def __init__(self, decoder, tokenizer, classifier_tokenizer):
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.classifier_tokenizer = classifier_tokenizer

    def get_rep_rate(self, sents):
        num_uni_tokens, num_tokens = 0, 0
        for sent in sents:
            tokens = sent.strip().split()
            num_uni_tokens += len(set(tokens))
            num_tokens += len(tokens)
        return 1. - num_uni_tokens * 1.0 / num_tokens

    def _evaluate(self, args, dev, return_pred=False, discriminator=None):

        logging.info("start evaluation")
        hyp = []
        ref = []
        tmp_loss = []
        rewards = []
        deltas = []
        articles = []

        for _, data_dev in enumerate(dev):
            l = get_prob(args, self.decoder, self.tokenizer, data_dev)
            tmp_loss.append(float(l.data.cpu().numpy()))

            decoded_sents = decode_batch(self.decoder, self.tokenizer, data_dev)
            for i, sent in enumerate(decoded_sents):
                hyp.append(" ".join(sent))
                ref.append(" ".join(data_dev["target_txt"][i].split()))
                articles.append(data_dev["input_txt"][i])
            if discriminator is not None:
                rewards.extend([r for r in get_reward(
                    self.classifier_tokenizer, decoded_sents, data_dev, discriminator)])
            if "deltas" in data_dev:
                deltas.extend([float(s)
                                        for s in data_dev["deltas"]])

        rouge_score = rouge(hyp, ref)
        logging.info("score: {}, ref repeatition rate: {}, prediction repeatition rate: {}".format(
            rouge_score, self.get_rep_rate(ref), self.get_rep_rate(hyp)))
        dev_loss = np.mean(tmp_loss)
        logging.info("dev loss: "+str(dev_loss))
        logging.info("rewards: "+str(sum(rewards) / len(rewards)))

        if return_pred:
            return float(sum(rewards) / len(rewards)), dev_loss, (hyp, ref, rewards, deltas, articles)
        else:
            return float(sum(rewards) / len(rewards)), dev_loss

    def predict_batch(self, batch):
        hyp, ref = [], []
        decoded_sents = decode_batch(self.decoder, self.tokenizer, batch)
        for i, sent in enumerate(decoded_sents):
            hyp.append(" ".join(sent))
            ref.append(batch["target_txt"][i])
        return hyp, ref