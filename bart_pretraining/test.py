from transformers import AutoTokenizer, BartForConditionalGeneration

# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

model = BartForConditionalGeneration.from_pretrained("/projects/bblr/smoorjani/control_tuning/pretrained_bart/bart-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")



ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

model = model.to("cuda")
for k, v in inputs.items():
    inputs[k] = v.to("cuda")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=150)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])