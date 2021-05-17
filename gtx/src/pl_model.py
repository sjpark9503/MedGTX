# Base pkgs
import math
import pytorch_lightning as pl
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
# Usr defined pkgs
from utils.model import ProbingEncModel
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
        return self.model(x).logits
    
    def training_step(self, batch, batch_idx):
        if self.global_step==0:
            notifier.critical("Here is the actual input of model")
            notifier.warning(batch)
        loss = self.model(**batch).loss
        self.log('train_loss', loss)
        self.samples_seen+=len(batch['input_ids'])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss, outputs.scores

    def validation_epoch_end(self, val_epoch_outputs):
        epoch_scores = [v[-1] for v in val_epoch_outputs]
        epoch_scores = torch.cat(epoch_scores)
        epoch_loss = sum([v[0] for v in val_epoch_outputs])
        cur_val_metric = epoch_scores.sum()/epoch_scores.size(0)
        if self.best_val_metric < cur_val_metric:
            self.best_val_metric = cur_val_metric
        self.log(f'valid_loss', epoch_loss/len(self.val_dataloader()))
        self.log(f'valid_acc', cur_val_metric)
        if self.global_step == 0:
            self.log(f'ZeroShot_acc', cur_val_metric)
        self.log(f'best_valid_acc', self.best_val_metric)

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss, outputs.scores
    
    def test_epoch_end(self, test_epoch_outputs):
        epoch_scores = [v[-1] for v in test_epoch_outputs]
        epoch_scores = torch.cat(epoch_scores)
        self.log(f'test_acc', epoch_scores.sum()/epoch_scores.size(0))

    def metric_Pre(self, outputs):

        return NotImplementedError()

    def metric_Re(self, outputs):

        return NotImplementedError()

    def metric_AdmPred(self, outputs):

        return NotImplementedError()

    def metric_ErrDetect(self, outputs):

        return NotImplementedError()

    def metric_Gen(self, outputs):

        return NotImplementedError()

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
            ),#ReduceLROnPlateau(optimizer),
            'interval': 'step',
            # 'monitor': 'train_loss',
            # 'frequency': 10,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}