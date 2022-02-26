import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Initial text for GPT2 model', required=True)
parser.add_argument('--length', type=int, help='Amount of new words added to input', default=50)
parser.add_argument('--model', type=str, help='Amount of new words added to input', default="/ocean/projects/cis210020p/moorjani/sensational_headline/test_trainer/checkpoint-6000")
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
baseline = GPT2LMHeadModel.from_pretrained('gpt2')
ours = GPT2LMHeadModel.from_pretrained(args.model)

inputs = tokenizer(args.input, return_tensors='pt')

baseline_output = baseline.generate(
    **inputs, 
    max_length=args.length, 
    num_beams=5, 
    early_stopping=True
)

our_output = ours.generate(
    **inputs, 
    max_length=args.length, 
    num_beams=5, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print("Baseline: " + tokenizer.decode(baseline_output[0], skip_special_tokens=True))
print("Ours: " + tokenizer.decode(our_output[0], skip_special_tokens=True))