# Base packages
import logging
import math
import os
from time import time
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import torch
import numpy as np

# Own implementation
from utils.parameters import parser
from utils.dataset import get_dataset
from utils.data_collator import NegativeSampling_DataCollator, Evaluation_DataCollator
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from model import LxmertForRanking
from trainer import Trainer

# From Huggingface transformers package
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    LxmertConfig,
    LxmertTokenizer,
    PreTrainedTokenizer,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d %H:%M",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = LxmertConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = LxmertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    logger.info(config)
    config.use_ce_pooler = True
    if model_args.tokenizer_name:
        tokenizer = LxmertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = LxmertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    model = LxmertForRanking.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    data_collator = Evaluation_DataCollator(tokenizer=tokenizer,
                                            kg_special_token_ids=config.kg_special_token_ids,
                                            task=training_args.task)


    model.to(training_args.device)
    model.eval()

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    test_dataset = get_dataset(data_args, 
                            tokenizer=tokenizer, 
                            token_type_vocab = config.token_type_vocab, 
                            test=True)
    logger.info(test_dataset[0])
    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     test_dataset=test_dataset
    # )

    # Evaluation
    top_k = training_args.top_k
    data_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=training_args.eval_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
    )
    db = dict()
    datas = [data for data in data_loader]

    for k in datas[0]:
        db[k] = torch.cat([data[k] for data in datas]).to(training_args.device)
    base_output_dir = training_args.output_dir
    for task in ['graph_retrieval', 'text_retrieval']:
        training_args.task = task
        training_args.output_dir = base_output_dir.replace('retrieval', task)

        sample_hits = list()
        sample_rank = list()

        with torch.no_grad():
            for positive_idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
                scores = list()
                for idx in range(len(test_dataset)//training_args.per_device_eval_batch_size+1):
                    start_idx = idx*training_args.per_device_eval_batch_size
                    if idx < (len(test_dataset)//training_args.per_device_eval_batch_size):
                        end_idx = (idx+1)*training_args.per_device_eval_batch_size
                    else:
                        end_idx = len(test_dataset)
                    if start_idx == end_idx:
                        continue

                    inputs = dict()
                    for k in db:
                        if training_args.task in ['text_retrieval', 'single_text_retrieval']:
                            if 'kg' in k:
                                inputs[k] = torch.stack([db[k][positive_idx]]*(end_idx-start_idx))
                            else:
                                inputs[k] = db[k][start_idx:end_idx]
                        else:
                            if 'kg' not in k:
                                inputs[k] = torch.stack([db[k][positive_idx]]*(end_idx-start_idx))
                            else:
                                inputs[k] = db[k][start_idx:end_idx]

                    outputs = model(**inputs)

                    scores+=outputs.pooled_logits[:,1].tolist()

                ranks = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

                if positive_idx in ranks[:top_k]:
                    sample_hits.append(1)
                else:
                    sample_hits.append(0)
                sample_rank.append(1/(ranks.index(positive_idx)+1))

        logger.info("Evaluation on Test set is done!")
        logger.info("*"*20)
        logger.info(f"Model : {training_args.output_dir.split('/')[-2]}_{training_args.output_dir.split('/')[-1]}")
        logger.info(f"Hits@{top_k} = {sum(sample_hits)/len(sample_hits)}\tMRR = {sum(sample_rank)/len(sample_rank)}")
        logger.info("*"*20)
        if not os.path.isdir(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        with open(os.path.join(training_args.output_dir,'result.txt'),'w') as h:
            h.write(f"Hits@{top_k} = {sum(sample_hits)/len(sample_hits)}\tMRR = {sum(sample_rank)/len(sample_rank)}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()