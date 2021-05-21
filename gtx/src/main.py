# Base pkgs
import os
import json
import pytorch_lightning as pl
# Usr defined pckgs
from pl_data import DataModule
from pl_model import GTXModel
from utils.parameters import parser
# Logging tool
from utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

# def get_deepspeed_config(path):
#     if path:
#         ds_config = pl.plugins.DeepSpeedPlugin(config=path)
#         return ds_config
#     else:
#         return None

def get_trainer_config(args):
    # configuration for `pl.Trainer`
    config = {
        "max_epochs":args.num_train_epochs,
        "precision":16 if args.fp16 else 32,
        "gpus":None if args.use_tpu else -1,
        "tpu_cores":8 if args.use_tpu else None,
        "accelerator": "ddp",
        "log_every_n_steps":None if args.use_tpu else 50,
        "val_check_interval":0.2 if args.task != "Pre" else 1.0,
    }
    if not args.do_eval:
        config["val_check_interval"]=1e10
        
    return config

def main():
    # Parse arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    pl.seed_everything(training_args.seed)

    # Call Logger
    wandb_config = dict()
    wandb_config.update(vars(training_args))
    wandb_config.update(vars(model_args))
    logger = pl.loggers.WandbLogger(config=wandb_config, project='NeurIPS2021', name=training_args.run_name, save_dir=None)

    # Call Model
    gtx = GTXModel(model_args, training_args)

    # Call data module
    data_module = DataModule(data_args, model_args, training_args, config=gtx.model.config)
    
    # Call Trainer
    trainer = pl.Trainer(
        **get_trainer_config(training_args),
        num_sanity_val_steps=4,
        logger=logger,
        # profiler='advanced',
        # fast_dev_run=10,
        # plugins=get_deepspeed_config(training_args.deepspeed),
    ) 
        
    # Train & Validation
    if training_args.do_train:
        # trainer.validate(model=gtx, datamodule = data_module)
        trainer.fit(gtx, data_module)
        if training_args.task == "Pre":
            gtx.save()
            data_module.save()
        
    # Test
    if training_args.do_eval:
        trainer.test()

if __name__ == "__main__":
    main()