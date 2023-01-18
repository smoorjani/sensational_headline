import numpy as np

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
    print(range_dict)
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
