import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    get_scheduler,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoConfig
)
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from dutils.config import USE_CUDA, get_args
from dutils.losses import get_loss
from dutils.function import Switch
from dutils.data_utils import prepare_data_seq, collate_fn
from dutils.evaluation import compute_metrics

import sys
sys.path.append("..")
from speed_control.models import SpeedRegressor


import os
from numpy import random
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class CustomTrainer(Trainer):
    def __init__(
        self,
        args,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        custom_args,
        discriminator,
        classifier_tokenizer,
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

        self.discriminator = discriminator
        self.classifier_tokenizer = classifier_tokenizer

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset= None):
        return self.eval_dataloader

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = get_loss(self.custom_args, inputs, model, self.tokenizer, self.discriminator,
                            self.classifier_tokenizer)

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

        torch.nn.utils.clip_grad_norm(model.parameters(), self.custom_args.max_grad_norm)
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = get_loss(self.custom_args, inputs, model, self.tokenizer, self.discriminator,
                            self.classifier_tokenizer)

        outputs = None
        if return_outputs:
            outputs = model(**inputs)

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    custom_args = get_args()
    custom_args.generator_name = 'distilgpt2'
    custom_args.discriminator_name = 'bart'

    custom_args.generator = 'Zohar/distilgpt2-finetuned-restaurant-reviews-clean'
    custom_args.discriminator_path = os.environ['PROJECT'] + "/control_tuning/speed_control/checkpoint_6_0_False_5e-05_2_0.8_3_0.001"

    # custom_args.total_steps = 20000
    # custom_args.max_q = max_q
    # setting max decoding length
    custom_args.max_r = 40

    os.environ['MASTER_PORT'] = '12370'

    discriminator = SpeedRegressor()
    discriminator.load_state_dict(torch.load(custom_args.discriminator_path))

    # sd = torch.load(custom_args.persuasivness_clasifier_path'])['state_dict
    # state_dict = {key.replace('module.','') : value for key, value in sd.items()}
    # discriminator.load_state_dict(state_dict)

    ckpt_name = (
            f"model_{custom_args.generator_name}_{custom_args.hidden_size}_{custom_args.discriminator_name}"
            f"_{custom_args.gamma}_{custom_args.total_steps}_{custom_args.batch_size}_{custom_args.optimizer}"
            f"_{custom_args.lr}_{custom_args.weight_decay}_{custom_args.max_grad_norm}_{custom_args.eps}"
        )

    checkpoint_path = os.path.join(
        (os.environ['PROJECT'] + custom_args.save_path),
        ckpt_name
    )

    tb_path = os.path.join((os.environ['PROJECT'] + custom_args.log_dir), ckpt_name)
    # if not os.path.exists(tb_path):
        # os.mkdir(tb_path)

    writer = SummaryWriter(log_dir=tb_path)
    callbacks = [transformers.integrations.TensorBoardCallback(writer)]

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=(custom_args.batch_size / torch.cuda.device_count()),
        save_steps=max(custom_args.total_steps // 5, 10000),
    #   gradient_accumulation_steps=8,
        num_train_epochs=custom_args.epochs,
        max_steps=custom_args.total_steps,
        learning_rate=custom_args.lr,
        weight_decay=custom_args.weight_decay,
        max_grad_norm=custom_args.max_grad_norm,
        deepspeed=custom_args.ds_config,
        fp16=True,
        # evaluation
        # evaluation_strategy="steps",
        # eval_steps=10,
        # logging to tensorboard
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=10,

    )

    # print(training_args)
    device = torch.device('cuda', custom_args.local_rank)

    print('Loading data...')
    train_dataloader, eval_dataloader, max_r, switch = prepare_data_seq(
        os.environ['PROJECT'] + custom_args.training_data,
        os.environ['PROJECT'] + custom_args.eval_data,
        custom_args.batch_size,
        thd=custom_args.thd
    )

    special_tokens = list(switch.values())

    # print(f'Dataloader len {len(train_dataloader)}, max_r {max_r}')

    print('Loading generative model...')
    config = AutoConfig.from_pretrained(
        custom_args.generator, output_hidden_states=True, output_attentions=True)
    config.pad_token_id = config.eos_token_id

    decoder = AutoModelWithLMHead.from_pretrained(custom_args.generator, config=config)
    tokenizer = AutoTokenizer.from_pretrained(custom_args.generator)
    tokenizer.pad_token = tokenizer.eos_token
    
    print('Adding special tokens and adjusting model')
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    decoder.resize_token_embeddings(len(tokenizer))

    discriminator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)
    
    optimizer = None
    if custom_args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params=decoder.parameters(), lr=training_args.learning_rate, weight_decay=custom_args.weight_decay)
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
        discriminator = discriminator.to(device)
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
        discriminator=discriminator,
        classifier_tokenizer=discriminator_tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    torch.cuda.empty_cache()
    transformers.logging.set_verbosity_info()
    
    if os.path.exists(checkpoint_path) and os.path.isfile(os.path.join(checkpoint_path, 'pytorch_model.bin')):
        train_result = trainer.train(checkpoint_path)
    else:
        train_result = trainer.train()
    trainer.save_model()
