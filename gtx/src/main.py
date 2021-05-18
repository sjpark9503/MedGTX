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

def get_deepspeed_config(path):
    if path:
        ds_config = pl.plugins.DeepSpeedPlugin(config=path)
        return ds_config
    else:
        return None

def get_trainer_config(args):
    callbacks = list()

    # Checkpointing Criteria
    callbacks.append(pl.callbacks.ModelCheckpoint(
        monitor='',
        dirpath=args.output_dir,
        save_top_k=1,
        filename='best',
        mode='min',
        )
    )  

    # Early stop Criteria
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="best_valid_Inter_acc",
        min_delta=0.01,
        patience=5,
        mode="max",
        # check_finite=True,
        # stopping_threshold=0.9
    )

    if args.shot == "full":
        callbacks.append(early_stop_callback)

    if args.tpu_id:
        tpu_core_id = 8
    else:
        tpu_core_id = None

    config = {
        "max_epochs":args.num_train_epochs if args.shot=="full" else 1,
        "max_steps": None if args.shot=="full" else int(20000/args.train_batch_size),
        "precision":16 if args.fp16 else 32,
        "gpus":-1 if args.tpu_id is None else None,
        "tpu_cores":tpu_core_id,
        "accelerator":"ddp" if len(os.environ["CUDA_VISIBLE_DEVICES"])>1 else None,
        "log_every_n_steps":10 if args.shot=="full" else 1,
        "callbacks":callbacks,
        "val_check_interval":0.1 if args.shot=="full" else 1,
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
    logger = pl.loggers.WandbLogger(config=wandb_config, entity="edlab_sjpark", project='CoNLL2021_0518', name=training_args.run_name, save_dir=None)

    # Call Model
    gtx = GTXModel(model_args, training_args)

    # Call data module
    data_module = DataModule(data_args, model_args, training_args, config=gtx.model.config)
    
    # Call Trainer
    trainer = pl.Trainer(
        **get_trainer_config(training_args),
        num_sanity_val_steps=4,
        logger=logger,
        plugins=get_deepspeed_config(training_args.deepspeed),
    ) 
        
    # Train & Validation
    if training_args.do_train:
        # trainer.validate(model=gtx, datamodule = data_module)
        trainer.fit(gtx, data_module)
        trainer.save_checkpoint("last_epoch.ckpt")
        
    # Test
    if training_args.do_test:
        trainer.test()

if __name__ == "__main__":
    main()