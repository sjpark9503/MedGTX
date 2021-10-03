# Base pkgs
import os
import math
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm.auto import tqdm
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from accelerate import DistributedType
# User defined pkgs
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from ..data_utils.dataset import get_dataset
from ..data_utils.data_collator import NodeClassification_DataCollator, NegativeSampling_DataCollator, UniLM_DataCollator, AdmLvlPred_DataCollator, ErrorDetection_DataCollator, Evaluation_DataCollator, TemporalPred_DataCollator
from .model import GTXForKGTokPredAndMaskedLM, GTXForRanking, GTXForAdmLvlPrediction, GTXForErrorDetection, GTXForGeneration, GTXForTemporalPred
from ..utils.metrics import metrics_for_tasks
# Transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
# Logging tool
from ..utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

class Trainer():
    def __init__(self, model_args, training_args, data_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        self.best_val_metric = None
        self.best_test_metric = None

        self.es_config = {
            "target":"valid_Inter_loss",
            "patience":2,
            "mode":"min",
            "delta":0,
        }
        self.es_count = 0

        self.samples_seen = 0
        self.few_shot_valid_interval = list(map(int,self.training_args.few_shot_valid_interval.split(',')))

    def create_dataset(self, accelerator):
        # Load Tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Set block size for padding & truncating inputs
        if self.data_args.block_size <= 0:
            self.data_args.block_size = tokenizer.model_max_length
        else:
            self.data_args.block_size = min(self.data_args.block_size, tokenizer.model_max_length)
        
        # Load dataset
        if self.training_args.do_train:
            train_dataset = get_dataset(
                self.data_args,
                tokenizer=tokenizer,
                token_type_vocab=self.config.token_type_vocab,
            )
            self.master_print(message=train_dataset[0], level='warning',accelerator=accelerator)
            eval_dataset = get_dataset(
                self.data_args,
                tokenizer=tokenizer,
                token_type_vocab = self.config.token_type_vocab,
                evaluate=True
            )
        else:
            train_dataset = None
            eval_dataset = None
            
        if self.training_args.do_eval:
            test_dataset = get_dataset(
                self.data_args, 
                tokenizer=tokenizer, 
                token_type_vocab = self.config.token_type_vocab,
                test=True
            )
        else:
            test_dataset = None
        return train_dataset, eval_dataset, test_dataset, tokenizer

    def prepare_model(self, tokenizer, accelerator):
       # Load configuration
        if self.model_args.config_name:
            config = AutoConfig.from_pretrained(self.model_args.config_name)
        elif self.model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path)
        else:
            config = CONFIG_MAPPING[self.model_args.model_type]()
            self.master_print(message="You are instantiating a new config instance from scratch.", level='warning',accelerator=accelerator)
        if self.training_args.task in ['AdmPred','ReAdm','NextDx', 'Death30', 'Death180', 'Death365']:
            config.use_ce_pooler = False
        else:
            config.use_ce_pooler = True
        config.KnowMix = self.training_args.knowmix

        # Load model
        MODEL_CLASSES = {
            "Pre":GTXForKGTokPredAndMaskedLM,
            "Re":GTXForRanking,
            "Gen":GTXForGeneration,
            "AdmPred":GTXForAdmLvlPrediction,
            "ErrDetect":GTXForErrorDetection,
            "ReAdm":GTXForTemporalPred,
            "NextDx":GTXForTemporalPred,
            "Death30":GTXForTemporalPred,
            "Death180":GTXForTemporalPred,
            "Death365":GTXForTemporalPred,
        }

        # Truncate some weights for Admpred
        if self.model_args.model_name_or_path:
            if self.training_args.task in ['AdmPred','ReAdm','NextDx', 'Death30', 'Death180', 'Death365']:
                ckpt_path = os.path.join(self.model_args.model_name_or_path, 'pytorch_model.bin')
                load_model_dict = torch.load(ckpt_path)
                modified_model_dict = load_model_dict.copy()
                for param in load_model_dict:
                    if 'multi_pooler' in param:
                        modified_model_dict.pop(param)
                torch.save(modified_model_dict, ckpt_path)

            model = MODEL_CLASSES[self.training_args.task].from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
            )
            self.master_print(message=f"Load pretrained parameters from {self.model_args.model_name_or_path}", level='critical',accelerator=accelerator)
        else:
            if "init" in config.KnowMix:
                lit2word = torch.load(config.lit2word_path)
                lit2word = self.tokenizer(lit2word, add_special_tokens=False, padding='max_length', max_length=64, return_token_type_ids=False)
            else:
                lit2word=None
            model = MODEL_CLASSES[self.training_args.task](config, lit2word=lit2word)
            self.master_print(message="Training new model from scratch", level='critical',accelerator=accelerator)

        if 'AdmPred' == self.training_args.task:
            db =  self.training_args.run_name.split('/')[4 if self.training_args.knowmix else 3].split('_')[-1]
            model.class_weight = torch.load(os.path.join(os.getcwd(),f'fixed_data/{db}/adm_class_weight'))
            self.master_print(message=f"Remove unused Weights in Pretrained model for AdmPred", level='critical',accelerator=accelerator)

        model.training_args = self.training_args
        
        return model

    def prepare_dataloader(self, train_dataset, valid_dataset, test_dataset, tokenizer, test_bsize=None):
        COLLATORS = {
            "Pre": NodeClassification_DataCollator,
            "Re":NegativeSampling_DataCollator,
            "Gen":UniLM_DataCollator,
            "AdmPred":AdmLvlPred_DataCollator,
            "ErrDetect":ErrorDetection_DataCollator,
            "ReAdm":TemporalPred_DataCollator,
            "NextDx":TemporalPred_DataCollator,
            "Death30":TemporalPred_DataCollator,
            "Death180":TemporalPred_DataCollator,
            "Death365":TemporalPred_DataCollator,
        }

        # if self.args.task == "ErrDetect":
        #     id2desc=torch.load(self.args.id2desc)
        #     id2desc={k:self.tokenizer(v, add_special_tokens=False, return_token_type_ids=False)['input_ids'] for k,v in id2desc.items()}
        # else:
        id2desc=None

        collator_args = {
            "tokenizer": tokenizer,
            "align": self.training_args.align,
            "n_negatives": self.training_args.n_negatives,
            "edge_cls": self.training_args.edge_cls,
            "kg_special_token_ids": self.config.kg_special_token_ids,
            "kg_size": self.config.vocab_size['kg'],
            "num_labels": self.config.num_labels,
            "label_domain": self.training_args.label_domain,
            "id2desc": id2desc if self.training_args.knowmix else None,
        }
        data_collator = COLLATORS[self.args.task](**{k:v for k,v in collator_args.items() if k in COLLATORS[self.args.task].__annotations__}, prediction=self.training_args.do_predict)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_args.train_batch_size,
            collate_fn=data_collator,
            drop_last=self.training_args.dataloader_drop_last,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
            shuffle=True,
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.training_args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
            shuffle=False)

        if test_bsize is None:
            if self.training_args.task == "Re":
                bsize = 1
            else:
                bsize = self.training_args.eval_batch_size
        else:
            bsize = test_bsize
        
        if self.training_args.task == "Re":
            test_collator = data_collator.copy()
            test_collator.n_negatives=0
        else:
            test_collator = data_collator

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=bsize,
            collate_fn=test_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
            shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader

    def prepare_optimizer(self, model):
        # Define optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = {"Adadelta":Adadelta, "Adagrad":Adagrad, "Adam":Adam, "AdamW":AdamW}[self.training_args.optimizer]
        if "Adam" in self.training_args.optimizer:
            optimizer_kwargs = {
                "betas": (self.training_args.adam_beta1, self.training_args.adam_beta2),
                "eps": self.training_args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.training_args.learning_rate
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer

    def get_lr_scheduler(self, data_loader, optimizer):
        # Define scheduler
        num_update_steps_per_epoch = len(data_loader) // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.training_args.max_steps > 0:
            max_steps = self.training_args.max_steps
        else:
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)

        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=max_steps,
            )
        return lr_scheduler

    def prepare_all(self, train_dataset, valid_dataset, test_dataset, tokenizer, accelerator, wandb_config=None):
        # Prepare logger
        if accelerator.is_main_process:
            import wandb
            self.logger = wandb.init(config=wandb_config, entity="kgtxt", project="ACL_MedGTX", group=self.training_args.run_name, name=f"{str(self.training_args.seed)}")
        # Prepare Data Loader
        train_dataloader, valid_dataloader, test_dataloader = self.prepare_dataloader(train_dataset, valid_dataset, test_dataset, tokenizer)
        # Prepare Model
        model = self.prepare_model(tokenizer, accelerator)
        # Prepare Optimizer
        optimizer = self.prepare_optimizer(model)
        # Wrap everything w/ Huggingface Accelerate
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)
        lr_scheduler = self.get_lr_scheduler(train_dataloader, optimizer)
        
        return model, optimizer, train_dataloader, valid_dataloader, test_dataloader, lr_scheduler

    def train(self, model, train_dataloader, valid_dataloader, accelerator, optimizer, lr_scheduler,tokenizer):
        for epoch in range(self.training_args.num_train_epochs):
            model.train()
            # Initialize global step
            if epoch == 0:
                global_step = 0
            # Validation on every epoch start
            if (global_step%(self.training_args.val_check_interval*len(train_dataloader))==0) and (epoch%self.training_args.check_val_every_n_epoch==0):
                early_stop = self.evaluate(model, valid_dataloader, accelerator, global_step=global_step, eval_mode="valid", tokenizer=tokenizer)
            # Early stopping & Keep best model
            if early_stop:
                break
            elif self.es_count==0:
                self.keep_or_save(model, accelerator, mode="keep")
            # Training loop
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc= f"Train [{epoch}]", leave=False):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / self.training_args.gradient_accumulation_steps
                # Log
                if accelerator.is_main_process and ((step+epoch*len(train_dataloader))%self.training_args.logging_steps==0):
                    self.logger.log(outputs.loss_dict)
                accelerator.backward(loss)
                # Optimizer step
                if step % self.training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += self.training_args.gradient_accumulation_steps 

    def evaluate(self, model, eval_dataloader, accelerator, global_step, eval_mode, tokenizer):
        model.eval()
        epoch_outputs = list()
        # Prepare negative pairs for retrieval
        if (self.training_args.task == "Re") and (eval_mode=="test"):
            negative_sampler = list()
            for negative_sample in eval_dataloader(128):
                negative_sampler.append({k:v.cuda() for k,v in negative_sample.items()})

        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader) ,disable=not accelerator.is_main_process, desc=f"{eval_mode.upper()}", leave=False):
            with torch.no_grad():
                outputs = model(**batch)
            metric = metrics_for_tasks(
                task=self.training_args.task,
                stage=eval_mode,
                batch=batch,
                outputs=outputs,
                loss_only= self.training_args.task in ['ReAdm', 'NextDx', 'Death30', 'Death180', 'Death365'],
                model=model if self.training_args.task == "Gen" else None,
                tokenizer=tokenizer,
                # current_epoch=self.current_epoch,
            )
            
            if self.training_args.task in ['ReAdm', 'NextDx', 'Death30', 'Death180', 'Death365']:
                metric_set = (metric, outputs.pooled_logits, batch['label'])
            else:
                metric_set = (metric,)
            # Gather the sharded evaluation results
            if accelerator.distributed_type == DistributedType.TPU:
                gathered_metric = accelerator.gather(metric_set)
            # Or not! 
            else:
                gathered_metric = metric_set
            epoch_outputs.append(gathered_metric)

        epoch_metrics = [x[0] for x in epoch_outputs]
        keys = epoch_metrics[0].keys()
        epoch_metrics = {k:torch.cat([val_epoch_metric[k].float() \
            if len(val_epoch_metric[k].size())>0 else val_epoch_metric[k].unsqueeze(0) \
                for val_epoch_metric in epoch_metrics]).float().mean() for k in keys}
        if self.training_args.task in ['ReAdm', 'NextDx', 'Death30', 'Death180', 'Death365']:
            epoch_metrics.update(
                metrics_for_tasks(
                    task=self.training_args.task,
                    stage="valid",
                    gt= torch.cat([x[2] for x in epoch_outputs],dim=0),
                    outputs=torch.cat([x[1] for x in epoch_outputs],dim=0),
                    model=None,
                    tokenizer=self.tokenizer,
                    current_epoch=self.current_epoch,
                )
            )
        if accelerator.is_main_process:
            self.logger.log(epoch_metrics)

        return EARLYSTOP_FLAG if eval_mode=="valid" else False

    def EarlyStopping(self, metrics, eval_mode, db_name):
        target = self.es_config['target']
        if target in metrics:
            if self.best_eval_metric[eval_mode][db_name] is None:
                self.best_eval_metric[eval_mode][db_name] = metrics[target]
            elif (self.best_eval_metric[eval_mode][db_name] < metrics[target]) and (self.es_config["mode"]=="max"):
                self.best_eval_metric[eval_mode][db_name] = metrics[target]
                self.es_count = 0
                return False
            elif (self.best_eval_metric[eval_mode][db_name] > metrics[target]) and (self.es_config["mode"]=="min"):
                self.best_eval_metric[eval_mode][db_name] = metrics[target]
                self.es_count = 0
                return False
            else:
                self.es_count += 1
                if self.es_count < self.es_config["patience"]:
                    return False
                else:
                    return True
        else:
            return False

    def keep_or_save(self, model, accelerator, mode=None):
        output_dir = self.training_args.output_dir
        # Wait for all distributed workers & unwrap model
        unwarpped_model = accelerator.unwrap_model(model)
        if mode == "keep":
            self.best_parameter = unwarpped_model.state_dict()
        elif mode == "save":
            accelerator.wait_for_everyone()
            unwarpped_model.load_state_dict(self.best_parameter)
            os.makedirs(output_dir, exist_ok=True)
            accelerator.save(unwarpped_model.state_dict(), os.path.join(output_dir,"pytorch_model.bin"))
            accelerator.save(unwarpped_model.config, os.path.join(output_dir,"config.json"))
            self.master_print(accelerator=accelerator,level="warning",message=f"Successfully saved model to {output_dir}")
        else:
            raise NotImplementedError()

        return model

    def master_print(self, message, level, accelerator=None):
        def _level_wise_print(message, level):
            if level == "warning":
                notifier.warning(message)
            elif level == "critical":
                notifier.critical(message)
            else:
                notifier.info(message)
        if accelerator is None:
            _level_wise_print(message=message,level=level)
        else:
            if accelerator.is_main_process:
                _level_wise_print(message=message,level=level)

    def save_decode_files(self, decode_outputs, output_dir, accelerator):
        assert self.training_args.task == "Gen"  # only for generation task
        assert 'eval_output/' in output_dir  # only for evaluation mode
        os.makedirs(output_dir, exist_ok=True)
        
        final_outputs = []
        if isinstance(decode_outputs, list):
            for decode_output in decode_outputs:
                gt_graph, gt_text, pred = decode_output
                data = {'graph': gt_graph, 'text': gt_text, 'pred': pred}
                final_outputs.append(data)

        import json
        final_outputs_fname = os.path.join(output_dir, 'test_output.pt')
        # with open(final_outputs_fname, 'w') as fout:
        #     json.dump(final_outputs, fout)
        accelerator.save(final_outputs, final_outputs_fname)
            
        self.master_print(accelerator=accelerator,level="warning",message=f"Save eval output files to {output_dir}")



