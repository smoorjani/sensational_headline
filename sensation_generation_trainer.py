import torch
import random
import numpy as np
import evaluate
import transformers
from transformers import (
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    get_scheduler,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from dutils.evaluation import postprocess_text
from dutils.config import USE_CUDA, get_args
from dutils.losses import get_tuning_loss, get_mle_loss
from dutils.function import Switch
from dutils.data_utils import prepare_data_seq, collate_fn, get_data

import ssl
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from word_embedding_measures.utils.embeddings import load_fasttext

import sys
sys.path.append("..")
from speed_control.models import SpeedRegressor


import os

torch.autograd.set_detect_anomaly(True)


class CustomTrainer(Trainer):
    def __init__(
        self,
        args,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        custom_args,
        discriminator_utils,
        direct_comp_utils,
        compute_metrics,
        callbacks
    ):
        super(CustomTrainer, self).__init__(
            args=args,
            model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )

        self.custom_args = custom_args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.discriminator_utils = discriminator_utils
        self.direct_comp_utils = direct_comp_utils

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset= None):
        return self.eval_dataloader

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
            
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), self.custom_args.max_grad_norm)
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device('cuda', self.custom_args.local_rank)
        inputs = {key: item.to(device) for key, item in inputs.items()}

        # out = decoder(
        if self.custom_args.control_method == 'emb':
            out = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=None,
                labels=inputs['labels'],
                control=inputs['deltas']
            )
        
        for k, v in inputs.items():
            print(f'{k}, {v.shape}, {v}')

        else:
            out = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=None,
                # decoder_input_ids=inputs['decoder_input_ids'],
                labels=inputs['labels']
            )

        loss = (1 - self.custom_args.gamma) * out.loss

        tuning_loss = 0
        if custom_args.gamma:
            tuning_loss, output_ids = get_tuning_loss(
                self.custom_args,
                inputs,
                model,
                self.tokenizer,
                self.discriminator_utils,
                self.direct_comp_utils
            )

            loss += self.custom_args.gamma * tuning_loss

        print(f'Total Loss: {loss}, MLE: {out.loss}, Tuning: {tuning_loss}')

        outputs = None
        if return_outputs:
            outputs = out
            outputs['generated'] = output_ids

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    custom_args = get_args()
    custom_args.generator = 'facebook/bart-base'
    # custom_args.generator = '/projects/bblr/smoorjani/control_tuning/pretrained_bart/bart-base/'
   
    custom_args.generator_name = 'sbart' if custom_args.control_method == 'emb' else custom_args.generator.split('/')[-1]
    custom_args.discriminator_name = 'bart' if 'bart' in custom_args.discriminator_path else 'roberta'

    # custom_args.generator = 'Zohar/distilgpt2-finetuned-restaurant-reviews-clean'
    # custom_args.generator = 'twigs/bart-text2text-simplifier' # 10000
    # custom_args.generator = 'philschmid/bart-large-cnn' # 15000
    
    # custom_args.discriminator_path = os.environ['PROJECT'] + "/control_tuning/speed_control/bart/checkpoint_6_0_False_5e-05_2_0.8_3_0.001"

    # custom_args.total_steps = 20000
    # custom_args.max_q = max_q
    # setting max decoding length
    custom_args.max_r = 40

    device = torch.device('cuda', custom_args.local_rank)
    # os.environ['MASTER_PORT'] = str(random.randint(12000, 13000))

    # seed the run
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    discriminator_utils, direct_comp_utils = None, None
    if custom_args.use_discriminator:
        print('Loading discriminator...')
        discriminator = SpeedRegressor('facebook/bart-base' if custom_args.discriminator_name == 'bart' else 'roberta-base')
        discriminator.load_state_dict(torch.load(custom_args.discriminator_path))
        discriminator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)
        if USE_CUDA:
            discriminator = discriminator.to(device)

        for param in discriminator.parameters():
            param.requires_grad = False

        discriminator_utils = (discriminator, discriminator_tokenizer)
    else:
        print('Loading fasttext model...')
        ft_model = load_fasttext(os.environ['PROJECT'] + custom_args.fasttext_model, limit=10000)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        stemmer = WordNetLemmatizer()
        en_stop = set(stopwords.words('english'))

        direct_comp_utils = (ft_model, stemmer, en_stop)



    # sd = torch.load(custom_args.persuasivness_clasifier_path'])['state_dict
    # state_dict = {key.replace('module.','') : value for key, value in sd.items()}
    # discriminator.load_state_dict(state_dict)

    ckpt_name = None
    if custom_args.use_discriminator:
        ckpt_name = (
            f"{custom_args.generator_name}_{custom_args.control_method}_{custom_args.hidden_size}_{custom_args.discriminator_name}"
            f"_{custom_args.gamma}_{custom_args.total_steps}_{custom_args.batch_size}_{custom_args.optimizer}"
            f"_{custom_args.lr}_{custom_args.weight_decay}_{custom_args.max_grad_norm}_{custom_args.eps}"
        )
    else:
        ckpt_name = (
            f"{custom_args.generator_name}_{custom_args.control_method}_{custom_args.hidden_size}_fasttext"
            f"_{custom_args.gamma}_{custom_args.total_steps}_{custom_args.batch_size}_{custom_args.optimizer}"
            f"_{custom_args.lr}_{custom_args.weight_decay}_{custom_args.max_grad_norm}_{custom_args.eps}"
        )

    ckpt_name += f'_{custom_args.notes}' if custom_args.notes else ''

    checkpoint_path = os.path.join(
        (os.environ['PROJECT'] + custom_args.save_path),
        ckpt_name
    )

    print(f'Writing to {checkpoint_path}')

    tb_path = os.path.join((os.environ['PROJECT'] + custom_args.log_dir), ckpt_name)
    # if not os.path.exists(tb_path):
        # os.mkdir(tb_path)

    writer = SummaryWriter(log_dir=tb_path)
    callbacks = [transformers.integrations.TensorBoardCallback(writer)]

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=(custom_args.batch_size / torch.cuda.device_count()),
        # save_steps=max(custom_args.total_steps // 5, min(custom_args.total_steps, 10000)),
        save_steps=10000,
    #   gradient_accumulation_steps=8,
        num_train_epochs=custom_args.epochs,
        max_steps=custom_args.total_steps,
        learning_rate=custom_args.lr,
        weight_decay=custom_args.weight_decay,
        max_grad_norm=custom_args.max_grad_norm,
        deepspeed=custom_args.ds_config,
        fp16=True,
        # evaluation
        evaluation_strategy="steps",
        eval_steps=500,
        eval_accumulation_steps=16,
        # logging to tensorboard
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=100,

    )

    # print(training_args)

    tokenizer = AutoTokenizer.from_pretrained(custom_args.generator)
    tokenizer.pad_token = tokenizer.eos_token

    print('Loading data...')
    train_dataloader, eval_dataloader, max_r, tokenizer = get_data(
        os.environ['PROJECT'] + custom_args.training_data,
        os.environ['PROJECT'] + custom_args.eval_data,
        custom_args.batch_size,
        tokenizer=tokenizer,
        thd=custom_args.thd,
        control_method=custom_args.control_method,
        # limit=1000
    )

    # print(f'Dataloader len {len(train_dataloader)}, max_r {max_r}')

    print('Loading generative model...')
    config = AutoConfig.from_pretrained(
        custom_args.generator, output_hidden_states=True, output_attentions=True)
    config.pad_token_id = config.eos_token_id
    if 'bart-large' in custom_args.generator:
        config.forced_bos_token_id = 0

    decoder = None
    if custom_args.control_method == 'emb':
        from transformers.models.bart_speed.modeling_sbart import SBartForConditionalGeneration
        decoder = SBartForConditionalGeneration.from_pretrained(custom_args.generator, config=config)
    else:
        if custom_args.gen_type == 'causal':
            decoder = AutoModelWithLMHead.from_pretrained(custom_args.generator, config=config)
        else:
            decoder = AutoModelForSeq2SeqLM.from_pretrained(custom_args.generator, config=config)

        print('Adjusting model for special tokens')
        decoder.resize_token_embeddings(len(tokenizer))

    sim_metrics = evaluate.combine(["rouge", "meteor"])
    bertscore = evaluate.load("bertscore")
    perplexity_metric = evaluate.load("perplexity", module_type="metric")

    def compute_metrics(eval_preds):
        # Predictions and labels are grouped in a namedtuple called EvalPrediction
        preds, labels = eval_preds

        preds[preds == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id

        if isinstance(preds, tuple):
            preds = preds[0]

        # if data_args.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)        
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # results: a dictionary with string keys (the name of the metric) and float
        # values (i.e. the metric values)
        result = sim_metrics.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result.update(bertscore.compute(predictions=decoded_preds, references=decoded_labels, model_type="distilbert-base-uncased"))
        if custom_args.gen_type == 'causal':
            result.update(perplexity_metric.compute(predictions=decoded_preds, model_id=custom_args.generator_name))
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        for eval_label in ["precision", "recall", "f1"]:
            result[eval_label] = np.mean(result[eval_label])

        return result
        
    optimizer = None
    if custom_args.optimizer.lower() == 'adam':
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model_params=decoder.parameters(), lr=training_args.learning_rate, weight_decay=custom_args.weight_decay)
        # optimizer = deepspeed.ops.adam.FusedAdam(params=decoder.parameters(), lr=training_args.learning_rate, weight_decay=custom_args.weight_decay)
    elif custom_args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=decoder.parameters(), lr=training_args.learning_rate, weight_decay=custom_args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params=decoder.parameters(), lr=training_args.learning_rate, weight_decay=custom_args.weight_decay)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=custom_args.total_steps
    )

    if USE_CUDA:
        decoder = decoder.to(device)
    
    dschf = HfDeepSpeedConfig(custom_args.ds_config)
    engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=decoder,
        config_params=custom_args.ds_config,
        optimizer=optimizer,
        training_data=train_dataloader,
        collate_fn=collate_fn
    )
    
    # torch.distributed.init_process_group(backend='nccl')
    # decoder = DataParallelModel(decoder, device_ids=[0,1])
    # decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[custom_args.local_rank']], output_device=custom_args['local_rank)
    # discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[custom_args.local_rank']], output_device=custom_args['local_rank)

    trainer = CustomTrainer(
        args=training_args,
        model=decoder,
        tokenizer=tokenizer,
        # optimizers=(optimizer, lr_scheduler), # doesn't work with deepspeed
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        custom_args=custom_args,
        discriminator_utils=discriminator_utils,
        direct_comp_utils=direct_comp_utils,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    torch.cuda.empty_cache()
    transformers.logging.set_verbosity_info()
    
    ignore_keys = ['logits', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions']
    if os.path.exists(checkpoint_path) and os.path.isfile(os.path.join(checkpoint_path, 'pytorch_model.bin')):
        train_result = trainer.train(checkpoint_path, ignore_keys_for_eval=ignore_keys)
    else:
        train_result = trainer.train(ignore_keys_for_eval=ignore_keys)
    trainer.save_model()
