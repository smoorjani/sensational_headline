import numpy as np
import torch
import torch.nn.functional as F

class Switch(dict):
    def __getitem__(self, item):
        for key in self.keys():                 # iterate over the intervals
            # if item in key:                     # if the argument is in that interval
            if key[0] <= item < key[1]:
                return super().__getitem__(key) # return its associated value
        raise KeyError(item)                    # if not in any interval, raise KeyError

def get_default_switch(deltas, num_bins=10, zero_thd=1e-4):

    range_dict = {
        (-zero_thd, zero_thd): '<SPEED_0>',
    }

    pos_deltas = sorted(list(filter(lambda x: x >= zero_thd, deltas)))
    neg_deltas = sorted(list(filter(lambda x: x < -zero_thd, deltas)))

    # handle case where there aren't an equal number of deltas for each side
    # note: this should not ideally be the case
    pos_ratio = len(pos_deltas) / (len(pos_deltas) + len(neg_deltas))
    num_pos_bins = int(pos_ratio * num_bins) + 1 # equivalent to math.ceil
    num_neg_bins = num_bins - num_pos_bins

    def create_ranges(range_dict, deltas, num_bins, multiplier):
        boundaries = [i for i in range(0, len(deltas), len(deltas) // (num_bins - 1))]

        dir_char = 'P' if multiplier == 1 else 'N'
        for i in range(len(boundaries)):
            if i+1 == len(boundaries):
                a = deltas[boundaries[i]]
                b = multiplier * float('inf')
                r = (a, b) if dir_char == 'P' else(b, a)
                range_dict[r] = f'<SPEED_{dir_char}{i+1}>'
            elif i+1 == 1:
                a = multiplier * zero_thd
                b = deltas[boundaries[i+1]]
                r = (a, b) if dir_char == 'P' else(b, a)
                range_dict[r] = f'<SPEED_{dir_char}{i+1}>'
            else:
                a = deltas[boundaries[i]]
                b = deltas[boundaries[i+1]]
                r = (a, b) if dir_char == 'P' else(b, a)
                range_dict[r] = f'<SPEED_{dir_char}{i+1}>'

        return range_dict

    range_dict = create_ranges(range_dict, pos_deltas, num_pos_bins, 1)
    neg_deltas.reverse()
    range_dict = create_ranges(range_dict, neg_deltas, num_neg_bins, -1)
    default_switch = Switch(range_dict)
    
    return default_switch

def harmonic_mean(r1, r2, beta):

    return (1. + beta ** 2) * (r1 * r2) / (beta ** 2 * r1 + r2) 

def cosine_similarity(vector1, vector2):
 
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)) + 1e-8)


def load_lexicons(lexicon_path):

    with open(lexicon_path+"/positive.txt", "r") as f:
        pos = [line.strip() for line in f.readlines()]

    with open(lexicon_path+"/negative.txt", "r") as f:
        neg = [line.strip() for line in f.readlines()]

    return set(pos + neg)

def load_va_lexicons(sentiment_lexicon_file):
    word2valence = {}
    word2arousal = {}
    with open(sentiment_lexicon_file, "r") as f:
        for line in f.readlines()[1:]:
            elements = line.strip().split(u",") # No.,Word,Valence_Mean,Valence_SD,Arousal_Mean,Arousal_SD,Frequency
            word = elements[1]
            word2valence[word] = float(elements[2])
            word2arousal[word] = float(elements[4])

    return {"word2valence": word2valence, "word2arousal": word2arousal}

def truncate_batch(batch, max_len=512):
    if batch['input_ids'].shape[-1] > max_len:
        for key, item in batch.items():
            batch_size, tokens = item.shape
            new_item = torch.zeros(batch_size, max_len)
            new_item = item[:, (tokens-max_len):]
            batch[key] = new_item
    return batch

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
