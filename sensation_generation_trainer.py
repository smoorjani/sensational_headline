import torch
from transformers import Trainer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer, TrainingArguments, get_scheduler

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from dutils.config import USE_CUDA, get_args
from models.losses import get_rl_loss
from models.sensation_scorer import PersuasivenessClassifier
from dutils.data_utils import prepare_data_seq
from dutils.parallel import DataParallelModel

from numpy import random
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class CustomTrainer(Trainer):
    def __init__(self, args, model, tokenizer, optimizers, train_dataloader, eval_dataloader, custom_args, sensation_model, classifier_tokenizer):
        super(CustomTrainer, self).__init__(args=args, model=model, tokenizer=tokenizer, optimizers=optimizers)
        self.custom_args = custom_args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.sensation_model = sensation_model
        self.expected_reward_layer = torch.nn.Linear(
            custom_args["hidden_size"], 1)

        self.classifier_tokenizer = classifier_tokenizer

        self.expected_rewards_loss = 0

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset= None):
        return self.eval_dataloader

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            # loss = self.compute_loss(model, inputs)
            _, loss, expected_reward_loss = get_rl_loss(self.custom_args, inputs, model, self.tokenizer, self.sensation_model,
                                                        self.classifier_tokenizer, self.expected_reward_layer, use_s_score=self.custom_args["use_s_score"])

        if self.args.n_gpu > 1:
            loss = loss.mean()
            expected_reward_loss = expected_reward_loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            expected_reward_loss = expected_reward_loss / \
                self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
            self.scaler.scale(expected_reward_loss).backward()
        elif self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            with amp.scale_loss(expected_reward_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
            expected_reward_loss = self.deepspeed.backward(
                expected_reward_loss)
        else:
            loss.backward()
            expected_reward_loss.backward()

        self.expected_rewards_loss += expected_reward_loss.data

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        _, loss, _ = get_rl_loss(self.custom_args, inputs, model, self.tokenizer, self.sensation_model,
                                 self.classifier_tokenizer, self.expected_reward_layer, use_s_score=self.custom_args["use_s_score"])

        outputs = None
        if return_outputs:
            outputs = model(**inputs)

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    custom_args = get_args()

    training_args = TrainingArguments("test_trainer", 
                                      per_device_train_batch_size=1,
                                      num_train_epochs=10,
                                      deepspeed="ds_config.json",
                                      fp16=True
                                    )
    # torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', custom_args['local_rank'])

    print('Loading data...')
    train_dataloader, eval_dataloader, max_q, max_r = prepare_data_seq(custom_args['training_data'], custom_args['eval_data'], custom_args['batch_size'], thd=custom_args['thd'])
    custom_args['max_q'] = max_q
    custom_args['max_r'] = 120

    print('Loading gpt model...')
    config = GPT2Config.from_pretrained(
        "gpt2", output_hidden_states=True, output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bert_tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased")
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = config.eos_token_id
    decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    print('Loading persuasiveness classifier...')
    sensation_model = PersuasivenessClassifier(bert_tokenizer.pad_token)
    # sensation_model.load_state_dict(torch.load(custom_args['persuasivness_clasifier_path']))
    # sensation_model.bert.resize_token_embeddings(len(vocab))

    # TODO: optimizers, RL and GPT
    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10000
    )
    # TODO: saving and loading

    if USE_CUDA:
        sensation_model = sensation_model.to(device)
        decoder = decoder.to(device)
    
    ds_config = './ds_config.json'
    dschf = HfDeepSpeedConfig(ds_config)
    engine = deepspeed.initialize(model=decoder, config_params=ds_config, optimizer=optimizer)
    # decoder = DataParallelModel(decoder, device_ids=[0,1])
    # decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[custom_args['local_rank']], output_device=custom_args['local_rank'])
    # sensation_model = torch.nn.parallel.DistributedDataParallel(sensation_model, device_ids=[custom_args['local_rank']], output_device=custom_args['local_rank'])

    trainer = CustomTrainer(
        args=training_args,
        model=decoder,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        custom_args=custom_args,
        sensation_model=sensation_model,
        classifier_tokenizer=bert_tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model()
