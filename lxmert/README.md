# BERT-pretraining
Scripts for pretraining LM based on [HuggingFace transformers](https://huggingface.co/transformers/)
### Requirements
- PyTorch >= 1.3.1
- transformers == 3.0.2
- wandb == 0.9.2 [wandb setup](https://docs.wandb.com/library/integrations/huggingface)

### Options
    run_language_modeling.py
            [--model_type MODEL_TYPE]
            [--config_name CONFIG_PATH OR NAME]
            [--tokenizer_name TOKENIZER_PATH OR NAME]
            [--train_data_file TRAIN_DATA_FILE_PATH]
            [--eval_data_file EVAL_DATA_FILE_PATH]
            [--line_by_line] [--mlm]
            [--mlm_probability MLM_PROBABILITY]
            [--plm_probability PLM_PROBABILITY]
            [--max_span_length MAX_SPAN_LENGTH]
            [--block_size BLOCK_SIZE] [--overwrite_cache]
            [--output_dir OUTPUT_DIR_PATH]
            [--overwrite_output_dir] [--do_train]
            [--do_eval] [--do_predict]
            [--evaluate_during_training]
            [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
            [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
            [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
            [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
            [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
            [--learning_rate LEARNING_RATE]
            [--weight_decay WEIGHT_DECAY]
            [--adam_epsilon ADAM_EPSILON]
            [--max_grad_norm MAX_GRAD_NORM]
            [--num_train_epochs NUM_TRAIN_EPOCHS]
            [--max_steps MAX_STEPS]
            [--warmup_steps WARMUP_STEPS]
            [--logging_dir LOGGING_DIR]
            [--logging_first_step]
            [--logging_steps LOGGING_STEPS]
            [--save_steps SAVE_STEPS]
            [--save_total_limit SAVE_TOTAL_LIMIT]
            [--no_cuda] [--seed SEED] [--fp16]
            [--fp16_opt_level FP16_OPT_LEVEL]
            [--local_rank LOCAL_RANK]
            [--tpu_num_cores TPU_NUM_CORES]
            [--tpu_metrics_debug] [--debug]
            [--dataloader_drop_last]
            [--eval_steps EVAL_STEPS]
            [--past_index PAST_INDEX]

### Usage
- Change model hyperparameters : add or modify ./config/[CONFIG_NAME]/config.json
- Data format : 1 sentence/line (if --line_by_line)

### Example (Bert)
```
CUDA_VISIBLE_DEVICES=0 python run_language_modeling.py --model_type='bert' --config_name='./config/bert-small' --tokenizer_name='bert-base-uncased' --train_data_file='data/test/wiki_train' --eval_data_file='data/test/wiki_test' --line_by_line --mlm --output_dir='pretrained_model' --do_train --do_eval --evaluate_during_training
```

### Example (Electra)
```
export HF=<PATH,.py> && CUDA_VISIBLE_DEVICES=<GPU_NUM> python $HF/run_discriminative_task.py --model_type=electra --config_name=google/electra-small-discriminator --tokenizer_name=<PATH,TOKENIZER> --train_data_file=<PATH,DATA> --mlm --output_dir=$HF/pretrained_electra --do_train --logging_dir=$HF/<PATH,LOGGING_DIR> --seed=1234 --per_device_train_batch_size=64 --fp16 --num_train_epochs=40 --learning_rate=5e-4 --warmup_steps=10000 --logging_steps=100 --save_steps=10000 --max_steps=1000000 --block_size=128
```

## Example (Glue)
```
export TASK=CoLA && export HF=/home/edlab/sjpark/ELECTRA-pretraining && CUDA_VISIBLE_DEVICES=2 python $HF/run_glue_org.py --config_name=google/electra-small-discriminator --model_name_or_path=$HF/pretrained_electra/checkpoint-end --tokenizer_name=bert-base-uncased --data_dir=$HF/glue_data/$TASK --overwrite_output_dir --output_dir=$HF/glue_benchmark --do_train --do_eval --do_predict --evaluate_during_training --task_name=$TASK --per_device_train_batch_size=32 --max_seq_length=128 --save_steps=1000000 --logging_steps=1000000 --num_train_epochs=10 --learning_rate=3e-5 --warmup_steps=81 --eval_steps=27
```
