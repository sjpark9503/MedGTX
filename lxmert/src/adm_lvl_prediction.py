# Base packages
import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from torch.utils.data import ConcatDataset
import torch

# Own implementation
from utils.parameters import parser
from utils.dataset import get_dataset
from utils.data_collator import NegativeSampling_DataCollator, AdmLvlPred_DataCollator, ErrorDetection_DataCollator
from model import LxmertForRanking, LxmertForKGTokPredAndMaskedLM, LxmertForAdmLvlPrediction, LxmertForErrorDetection
from trainer import Trainer

# From Huggingface transformers package
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    LxmertConfig,
    LxmertTokenizer,
    PreTrainedTokenizer,
    # Trainer,
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

    config.use_ce_pooler=False
    config.pretrained_kg_embedding=""
    model = LxmertForAdmLvlPrediction.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    logger.info(config)
    #model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    # train_dataset = get_dataset(data_args,
    #                             tokenizer=tokenizer,
    #                             token_type_vocab=config.token_type_vocab,
    #                             )
    # logger.info(train_dataset[0])
    # eval_dataset = get_dataset(data_args,
    #                             tokenizer=tokenizer,
    #                             token_type_vocab=config.token_type_vocab,
    #                             evaluate=True,
    #                             )
    test_dataset = get_dataset(data_args,
                                tokenizer=tokenizer,
                                token_type_vocab=config.token_type_vocab,
                                test=True
                                ) if training_args.do_eval else None

    data_collator = AdmLvlPred_DataCollator(tokenizer=tokenizer,
                                                num_kg_labels=config.num_kg_labels,
                                                kg_special_token_ids=config.kg_special_token_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        #eval_data_collator=eval_data_collator,
        #train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        test_dataset=test_dataset
    )
    trainer.args.top_k = 3
    outputs = trainer.predict(test_dataset)
    trainer.args.top_k = 5
    outputs = trainer.predict(test_dataset)
    # if not os.path.isdir(training_args.output_dir):
    #     os.makedirs(training_args.output_dir)
    # with open(os.path.join(training_args.output_dir,'result.txt'),'w') as h:
    #     h.write(f"P@{training_args.top_k} = {outputs.metrics['eval_P@{training_args.top_k}']}")
        
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()