import sys
# Usr defined pckgs
from src.configs.parameters import parser
from transformers import set_seed
# Logging tool
from src.utils.notifier import logging, log_formatter
from src.core.trainer import Trainer
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())
# Load Huggingface accelerate
from accelerate import Accelerator, DistributedType

def main():
    accelerator = Accelerator(split_batches=True)

    # Parse arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.val_check_interval = 1.0 if training_args.task in ["Pre"] else 0.5
    if not training_args.do_eval:
        training_args.val_check_interval=1e10
    if training_args.task == "Gen":
        training_args.val_check_interval=1.0
        training_args.check_val_every_n_epoch=5    
    data_args.knowmix = training_args.knowmix
    data_args.task = training_args.task

    # Set seed
    set_seed(training_args.seed)

    # Call Logger
    wandb_config = dict()
    wandb_config.update(vars(training_args))
    wandb_config.update(vars(model_args))

    # Call Trainer
    trainer = Trainer(model_args, training_args, data_args)
    # Create dataset on each device
    train_dataset, valid_dataset, test_dataset, tokenizer = trainer.create_dataset(accelerator)
    # Prepare data loader, model and optimizer (include lr scheduler)
    model, optimizer, train_dataloader, valid_dataloaders, test_dataloaders, lr_scheduler = trainer.prepare_all(train_dataset, valid_dataset, test_dataset, tokenizer, accelerator, wandb_config)

    # Train model
    if training_args.do_train:
        trainer.train(model, train_dataloader, valid_dataloaders, accelerator, optimizer, lr_scheduler)
        trainer.keep_or_save(model, accelerator, mode="save")
    # Evaluate model
    if training_args.do_test and ((training_args.task != "Gen") or (training_args.task !="Re")):
        trainer.evaluate(model, test_dataloaders, accelerator, global_step=-1, eval_mode="test")
    if accelerator.is_main_process:
        trainer.master_print(level="critical", message="Done! Terminate process")
    # Terminate DDP manually
    accelerator.wait_for_everyone()
    sys.exit("Cleaning up...") 
if __name__ == "__main__":
    main()