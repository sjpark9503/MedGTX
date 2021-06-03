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
    callbacks = list()

    # Checkpointing Criteria
    # callbacks.append(pl.callbacks.ModelCheckpoint(
    #     monitor='',
    #     dirpath=args.output_dir,
    #     save_top_k=1,
    #     filename='best',
    #     mode='min',
    #     )
    # )  

    monitoring_target = {
        "Pre":None,
        "Re":"valid_acc",
        "AdmPred":"valid_P@1",
        "ErrDetect":"valid_R@1",
        "Gen":"valid_lm_acc",
    }

    # Early stop Criteria
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitoring_target[args.task],
        min_delta=0.001 if args.task != "Gen" else 0.005,
        patience=3,
        verbose=True,
        mode="max",
        # check_finite=True,
        # stopping_threshold=0.9
    )
    
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(
        logging_interval="step"
    )

    if args.task != "Pre":
        callbacks.append(early_stop_callback)
        
    if args.task == "Gen":
        callbacks.append(lr_monitor_callback)

    if args.use_tpu:
        tpu_core_id = 8
    else:
        tpu_core_id = None

    config = {
        "max_epochs":args.num_train_epochs,
        "precision":16 if args.fp16 else 32,
        "gpus":None if args.use_tpu else -1,
        "tpu_cores":tpu_core_id,
        "accelerator": "ddp",
        "log_every_n_steps":None if args.use_tpu else 50,
        "callbacks":callbacks,
        # "check_val_every_n_epoch":50,
        "val_check_interval":1.0 if args.task in ["Pre"] else 0.2,
    }
    if not args.do_eval:
        config["val_check_interval"]=1e10
        
    if args.task == "Gen":
        config["val_check_interval"]=1.0
        config["check_val_every_n_epoch"]=5

    return config

def main():
    # Parse arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.knowmix = training_args.knowmix
    data_args.task = training_args.task
    notifier.warning(training_args.task in ["Pre", "ErrDetect"])

    # Set seed
    pl.seed_everything(training_args.seed)

    # Call Logger
    wandb_config = dict()
    wandb_config.update(vars(training_args))
    wandb_config.update(vars(model_args))
    logger = pl.loggers.WandbLogger(config=wandb_config,
                                    entity='kgtxt',
                                    project='NeurIPS2021',
                                    name=training_args.run_name,
                                    save_dir=None)

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
        # TODO: save models
        elif training_args.task == "Gen":
            gtx.save()
            # data_module.save()
        
    # Test
    if training_args.do_eval:
        if not training_args.do_train:
            data_module.prepare_data()
            data_module.setup('test')
        trainer.test(model=gtx, datamodule=data_module)

if __name__ == "__main__":
    main()