import os
import glob
import torch
import logging
import argparse
import pandas as pd

from bert_score import score
from datasets import load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, BartForConditionalGeneration
from transformers.models.bart_speed.modeling_sbart import SBartForConditionalGeneration

from dutils.data_utils import read_langs
from dutils.function import get_default_switch

BASE_DIR = '/projects/bblr/smoorjani/control_tuning'

MODEL_BASE_DIR = BASE_DIR + "/models"
DATA_BASE_DIR = BASE_DIR + "/dataset"
GENERATIONS_BASE_DIR = BASE_DIR + "/generations"

MODELS_TO_USE = [
    'experiment1'
]

SUPPORTED_METRICS = ['rouge', 'meteor', 'perplexity', 'bertscore']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=float, help='Number of tokens to take from each input', default=0.2)
    parser.add_argument('--length', type=int, help='Amount of new words added to input', default=100)
    parser.add_argument('--device', type=str, help='Device to inference on', default='cuda')

    parser.add_argument('--temperature', type=float, help='Temperature for decoding', default=0.6)
    parser.add_argument('--beams', type=int, help='Num beams for decoding', default=1)
    parser.add_argument('--repetition_penalty', type=float, help='Penalty for repetitive tokens', default=1.3)
    parser.add_argument('--top_k', type=int, help='Top-k for decoding', default=0)
    parser.add_argument('--top_p', type=float, help='Top-p for decoding', default=0)

    parser.add_argument('--train_file', type=str, help='File with train sentences/inps to get switch', default='/train.txt')
    parser.add_argument('--test_file', type=str, help='File with test sentences/inps', default='/eval.txt')
    parser.add_argument('--limit', type=int, help='Num of samples to use', default=1000)
    parser.add_argument('--logging_file', type=str, help='File to log to', default='./evaluation_log')

    parser.add_argument('--arch', type=str, help='Model architecture', default='facebook/bart-base')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default="")
    args = parser.parse_args()
    return args

def log_and_print(msg, logger):
    # print(msg)
    logger.info(msg)

def read_test_set(args):
    with open(DATA_BASE_DIR + args.test_file) as f:
        lines = f.readlines()

    inps = []
    targets = []
    deltas = []
    for i, line in enumerate(lines):
        if args.limit > 0 and i == args.limit:
            break
        # remove for memorability
        inp, s1, target, s2, target_delta = line.split('\t')

        inps.append(inp.strip())
        targets.append(target.strip())
        deltas.append(float(target_delta))

    return inps, targets, deltas

def generate_predictions(args, inps, deltas, model, tokenizer, switch, logger):
    log_and_print("Generating predictions!", logger)
    outputs = []
    for inp, delta in zip(inps, deltas):
        if switch:
            special_token = switch[float(delta)]
            inp = special_token + " " + inp

        inputs = tokenizer(inp, return_tensors='pt').to(args.device)
        output = None
        if not switch:
            output = model.generate(
                **inputs, 
                max_length=args.length, 
                num_beams=args.beams, 
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p if 0. < args.top_p < 1.  else 1.0,
                top_k=args.top_k if args.top_k > 0 else 50,
                control=torch.tensor(delta), # handled by model_specific_kwargs
            )
        else:
            output = model.generate(
                **inputs, 
                max_length=args.length, 
                num_beams=args.beams, 
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p if 0. < args.top_p < 1.  else 1.0,
                top_k=args.top_k if args.top_k > 0 else 50,
            )

        output = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(output)
        
    return outputs

def get_metric(args, references, metric_name='rouge', predictions=None, model_id=None):
    assert metric_name in SUPPORTED_METRICS
    metric = load_metric(metric_name)

    print(len(predictions))

    if 'perplexity' == metric_name:
        assert model_id is not None
        return {
            'perplexity': metric.compute(input_texts=references, model_id=model_id, device=args.device)['mean_perplexity']
        }
    elif 'rouge' == metric_name:
        assert predictions is not None
        rouge_scores = metric.compute(predictions=predictions, references=references)
        rouge_scores = {k: v.high for k, v in rouge_scores.items()}

        split_rouge_scores = {}
        for key, val in rouge_scores.items():
            split_rouge_scores[key + '_P'] = val.precision
            split_rouge_scores[key + '_R'] = val.recall
            split_rouge_scores[key + '_F'] = val.fmeasure
        return split_rouge_scores
    elif 'bertscore' == metric_name:
        P, R, F1 = score(predictions, references, lang="en")
        return {
            'bertscore_P': P.mean(),
            'bertscore_R': R.mean(),
            'bertscore_F1': F1.mean(),
        }
    else:
        assert predictions is not None
        return metric.compute(predictions=predictions, references=references)

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename=os.path.join(GENERATIONS_BASE_DIR, args.logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger('evaluation_logger')

    inps, targets, deltas = read_test_set(args)
    

    thd = 0.0
    d_train, _ = read_langs(DATA_BASE_DIR + args.train_file, thd)
    switch = get_default_switch([item['s'] for item in d_train])

    special_tokens = list(switch.values())

    MODELS_TO_USE = args.model_name if len(args.model_name) else glob.glob(MODEL_BASE_DIR + '/*')
    model_scores = []
    for model_name in MODELS_TO_USE:
        # if '50000' in model_name:
        #     continue
        # if '_0.0_' not in model_name:
        #     continue

        log_and_print('Adding special tokens to tokenizer', logger)
        tokenizer = AutoTokenizer.from_pretrained(args.arch, padding_side = 'left')
        tokenizer.padding_side = 'left'
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        predictions = None
        prediction_path = os.path.join(
            GENERATIONS_BASE_DIR,
            model_name.replace(MODEL_BASE_DIR, '').replace('/', '_')[1:] + \
             f'_gen_{args.beams}_{args.temperature}_{args.repetition_penalty}_{args.top_k}_{args.top_p}' + '.txt'
        )

        if os.path.exists(prediction_path):
            log_and_print(f'{model_name} predictions already exist!', logger)

            with open(prediction_path, 'r') as f:
                predictions = f.readlines()
            predictions = [p for i, p in enumerate(predictions) if i % 2 == 1 and i < len(targets) * 2]

        else:
            log_and_print(model_name + '\n', logger)

            try:
                model = None
                if 'sbart' in model_name:
                    model = SBartForConditionalGeneration.from_pretrained(os.path.join(MODEL_BASE_DIR, model_name)).to(args.device)
                else:
                    model = BartForConditionalGeneration.from_pretrained(os.path.join(MODEL_BASE_DIR, model_name)).to(args.device)
            except OSError:
                log_and_print(f"Can't load this model right now.", logger)
                continue
            
            predictions = generate_predictions(args, inps, deltas, model, tokenizer, switch, logger) if 'sbart' not in model_name \
                else generate_predictions(args, inps, deltas, model, tokenizer, None, logger)

            with open(prediction_path, 'w') as f:
                for inp, pred in zip(inps, predictions):
                    f.write(inp.replace('\n', '').strip() + '\n' + pred.replace('\n', '').strip() + '\n')

            del model

        scores = {'model': model_name}
        for metric in SUPPORTED_METRICS:
            try:
                results = get_metric(args, targets, metric, predictions, os.path.join(MODEL_BASE_DIR, model_name))
                log_and_print(results, logger)

                scores.update(results)
            except OSError:
                log_and_print(f"Can't load this model right now to calculate {metric}.", logger)
        model_scores.append(scores)
    
    pd.DataFrame(model_scores).to_csv('scores.csv', mode='a')
    log_and_print(model_scores, logger)

def postprocess_preds(predictions, inp_blacklist, blacklist, strict=False):
    new_preds = []
    for pred in predictions:
        for phrase in blacklist:
            pred = pred.replace(phrase, '')

        pred = pred.strip().replace('\n', ' ') + '\n'
        new_preds.append(pred)
    return new_preds