# Base pkgs
import os
import math
import time
import pytorch_lightning as pl
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
# Usr defined pkgs
from model import GTXForKGTokPredAndMaskedLM, GTXForRanking, GTXForAdmLvlPrediction, GTXForErrorDetection, GTXForGeneration
from utils.metrics import metrics_for_tasks
# Transformers
from transformers import (
    CONFIG_MAPPING,
    AutoModel,
    AutoModelForMaskedLM,
    AutoConfig,
)
# Logging tool
from utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

class GTXModel(pl.LightningModule):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args

        self.best_val_metric = -1e10
        self.best_test_metric = -1e10

        # Load configuration
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            notifier.warning("You are instantiating a new config instance from scratch.")
        if 'Adm' == training_args.task:
            config.use_ce_pooler = False
        else:
            config.use_ce_pooler = True

        # Load model
        MODEL_CLASSES = {
            "Pre":GTXForKGTokPredAndMaskedLM,
            "Re":GTXForRanking,
            "Gen":GTXForKGTokPredAndMaskedLM,
            "AdmPred":GTXForAdmLvlPrediction,
            "ErrDetect":GTXForErrorDetection,
        }
        if model_args.model_name_or_path:
            notifier.critical("Load pretrained parameters")
            self.model = MODEL_CLASSES[training_args.task].from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else:
            notifier.critical("Training new model from scratch")
            self.model = MODEL_CLASSES[training_args.task](config)

        self.model.training_args = training_args
        
        notifier.warn("## Model Configuration ##")
        notifier.warn(self.model.config)

    def forward(self, x):
        return self.model(x)
    
    # def on_training_epoch_start(self):
        # batch = next(self.train_dataloader())

    # def on_before_zero_grad(self,out):
    #     self.t1 = time.time()

    # def on_after_backward(self):
    #     self.t2 = time.time()

    # def on_train_batch_end(self, out, out2, out3, out4):
    #     self.t3 = time.time()
    #     if self.local_rank in [1,2,3]:
    #         notifier.warning(f"Backward Time : {self.t2-self.t1:3f}s")
    #         notifier.warning(f"Opt. Step Time : {self.t3-self.t2:3f}s")

    def training_step(self, batch, batch_idx):
        if (self.global_step==0) and (self.local_rank==1):
            notifier.critical("Here is the actual input of model")
            notifier.warning(batch)

        outputs = self.model(**batch)

        self.log_dict(outputs.loss_dict, on_step=False if self.training_args.use_tpu else True, on_epoch=True if self.training_args.use_tpu else False)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        metrics = metrics_for_tasks(task=self.training_args.task, stage="valid", batch=batch, outputs=outputs)
        return metrics

    def validation_epoch_end(self, val_epoch_outputs):
        keys = val_epoch_outputs[0].keys()
        epoch_metrics = {k:torch.cat([val_epoch_output[k].float() if len(val_epoch_output[k].size())>0 else val_epoch_output[k].unsqueeze(0) for val_epoch_output in val_epoch_outputs]).mean() for k in keys}
        self.log_dict(epoch_metrics)
        return epoch_metrics

    def test_step(self, batch, batch_idx):
        metrics = dict()
        if self.training_args.task == "Re":
            top_k = self.training_args.top_k
            for label_domain in ["graph", "text"]:
                for negative_sample in self.test_dataloader(batch_size=128):
                    scores = list()
                    for k in batch:
                        if label_domain == "text":
                            if 'kg' in k:
                                batch[k] = torch.stack([batch[k],]*negative_sample.size(0))
                            else:
                                batch[k] = negative_sample[k]
                        else:
                            if 'kg' not in k:
                                batch[k] = torch.stack([batch[k],]*negative_sample.size(0))
                            else:
                                batch[k] = negative_sample[k]

                    outputs = self.model(**batch)
                    scores.append(outputs.pooled_logits[:,1])

                scores = torch.cat(scores)
                hit = (scores.topk(top_k).indices==batch_idx).sum()
                recrank = 1/((scores.argsort(descending=True)==batch_idx).nonzero()+1)
                metrics[f"{label_domain}_Hits@{top_k}"] = hit
                metrics[f"{label_domain}_MRR"] = recrank

        else:
            outputs = self.model(**batch)
            metrics = metrics_for_tasks(task=self.training_args.task, stage="test", batch=batch, outputs=outputs)

        return metrics
    
    def test_epoch_end(self, test_epoch_outputs):
        keys = test_epoch_outputs[0].keys()
        epoch_metrics = {k:torch.cat([test_epoch_output[k] if len(test_epoch_output[k].size())!=0 else test_epoch_output[k].unsqueeze(0) for test_epoch_output in test_epoch_outputs]).mean() for k in keys}
        self.log_dict(epoch_metrics)
        return epoch_metrics

    def configure_optimizers(self):
        # Define optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
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

        # Define scheduler
        train_dataloader = self.train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.training_args.max_steps > 0:
            max_steps = self.args.max_steps
        else:
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)

        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=max_steps,
            ),
            'interval': 'step',
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def save(self):
        output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if (self.training_args.use_tpu and self.local_rank == 0) or not self.training_args.use_tpu:
            notifier.warning(f"Save model to {output_dir}")
            self.model.save_pretrained(output_dir)
            
    def configure_callbacks(self):
        # model_ckpt_callback = pl.callbacks.ModelCheckpoint(
        #             monitor='',
        #             dirpath=self.training_args.output_dir,
        #             save_top_k=1,
        #             filename='best',
        #             mode='min',
        # )
        
        if self.training_args.task == "Pre":
            return None 
        
        elif self.training_args.task == "Gen":
            monitor_metric = 'valid_lm_acc'
            early_stop_callback = pl.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    min_delta=0.01,
                    patience=3,
                    mode="max",
                    # check_finite=True,
                    # stopping_threshold=0.9
            )
            lr_monitor_callback = pl.callbacks.LearningRateMonitor(
                logging_interval="step",
            ) 
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor=monitor_metric,
                min_delta=0.01,
                patience=3,
                mode="max",
                # check_finite=True,
                # stopping_threshold=0.9
            )
            
            return [lr_monitor_callback, early_stop_callback]
        
        else:
            early_stop_callback = pl.callbacks.EarlyStopping(
                    monitor='val_acc',
                    min_delta=0.01,
                    patience=3,
                    mode="max",
                    # check_finite=True,
                    # stopping_threshold=0.9
            )
            return [early_stop_callback]