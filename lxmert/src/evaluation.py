# Base packages
import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from torch.utils.data import ConcatDataset
import torch
import numpy as np

# Own implementation
from utils.parameters import parser
from utils.dataset import get_dataset
from utils.data_collator import NegativeSampling_DataCollator
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

    config = LxmertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = LxmertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    if training_args.task in ['binary_retrieval', 'triplet_retrieval']:
        model = LxmertForRanking.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        data_collator = Evaluation_DataCollator(tokenizer=tokenizer,
                                                kg_special_token_ids=config.kg_special_token_ids,
                                                NCE=False)
    elif training_args.task in ['generation']: 
        model = LxmertForKGTokPredAndMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise NotImplementedError("Not implemented task: %s", training_args.task)
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

    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     test_dataset=test_dataset
    # )

    # Evaluation
    if training_args.task in ['binary_retrieval']:
        tok_k = 10
        data_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=training_args.eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )
        sample_hits = list()
        sample_rank = list()
        with torch.no_grad():
            for positive_idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
                scores = list()
                positive_sample = data_collator([test_dataset[positive_idx],]*training_args.eval_batch_size)
                for inputs in data_loader:
                    for k, v in inputs:
                        if 'kg' not in k:
                            inputs[k] = positive_sample[k]
                        inputs[k] = inputs[k].to(training_args.device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    scores.append((outputs.cross_relationship_score)[:,1].tolist())
                ranks = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                if positive_idx in ranks[:top_k]:
                    sample_hits.append(1)
                else:
                    sample_hits.append(0)
                sample_rank = 1/ranks[positive_idx]
        logger.info("Evaluation on Test set is done!")
        logger.info("*"*20)
        logger.info(f"Hits@{tok_k} = {sum(sample_hits)/len(sample_hits)}\tMRR = {sum(sample_rank)/len(sample_rank)}")
        logger.info("*"*20)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()