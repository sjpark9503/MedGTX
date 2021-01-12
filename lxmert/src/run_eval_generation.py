"""
Ver 0.3 for KG-LXMERT
"""
# Base packages
import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

# Own implementation
from utils.parameters import parser
from utils.dataset import get_dataset
from utils.data_collator import NegativeSampling_DataCollator
# from model import LxmertForRanking
from model import LxmertForKGTokPredAndMaskedLM
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

    if model_args.config_name:
        config = LxmertConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = LxmertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = LxmertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = LxmertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    if ((config.num_attention_heads % config.num_relations) != 0) and config.gcn and ('Multi' in training_args.output_dir):
        raise ValueError(
            "# attentions heads must be divisible by # relations"
        )

    if model_args.model_name_or_path:
        from model import LxmertForGeneration
        model = LxmertForGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            tokenizer=tokenizer
        )
    else:
        raise ValueError("Cannot Scratch Training")

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

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, kg_pad=config.kg_special_token_ids["PAD"]) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, kg_pad=config.kg_special_token_ids["PAD"], evaluate=True)
        if training_args.do_eval
        else None
    )
    # test_dataset = (
    #     get_dataset(data_args, tokenizer=tokenizer, kg_pad=config.kg_special_token_ids["PAD"], test=True)
    #     if training_args.do_eval
    #     else None
    # )
    
    
    if training_args.task == 'generation':
        from utils.data_collator import UniLM_DataCollator
        data_collator = UniLM_DataCollator(tokenizer=tokenizer,
                                           kg_special_token_ids=config.kg_special_token_ids,
                                           kg_size=config.vocab_size['kg'],
                                           prediction=True) # for generation
    else:
        raise NotImplementedError("Not implemented task")

    # train dataset
    if training_args.do_train and data_args.train_data_file:
        train_dataloader = DataLoader(train_dataset, 
                                      sampler=SequentialSampler(train_dataset),
                                    #   batch_size=training_args.per_device_train_batch_size,
                                      batch_size=64,
                                      collate_fn=data_collator,
                                      drop_last=training_args.dataloader_drop_last,
                                      num_workers=training_args.dataloader_num_workers,
                                      pin_memory=True
                                      )
        model.to(training_args.device)
        
        results = {}
        for idx, inputs in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Step'):
            # from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(training_args.device)
            
            with torch.no_grad():
                prd_text = model(**inputs).cpu().numpy()
                org_text_ids = inputs['lang_input_ids'].cpu().numpy()
                org_kg_ids = inputs['kg_input_ids'].cpu().numpy()
                
                items = {'prd_text': prd_text,
                            'org_text_ids': org_text_ids,
                            'org_kg_ids': org_kg_ids}
                
            results[idx] = items
            
            
    # eval dataset
    if training_args.do_eval and data_args.eval_data_file:
        eval_dataloader = DataLoader(eval_dataset, 
                                      sampler=SequentialSampler(eval_dataset),
                                    #   batch_size=training_args.per_device_train_batch_size,
                                      batch_size=64,
                                      collate_fn=data_collator,
                                      drop_last=training_args.dataloader_drop_last,
                                      num_workers=training_args.dataloader_num_workers,
                                      pin_memory=True
                                      )
        model.to(training_args.device)
        
        results = {}
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='Step'):
            # from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(training_args.device)
            
            with torch.no_grad():
                prd_text = model(**inputs).cpu().numpy()
                org_text_ids = inputs['lang_input_ids'].cpu().numpy()
                org_kg_ids = inputs['kg_input_ids'].cpu().numpy()
                
                items = {'prd_text': prd_text,
                         'org_text_ids': org_text_ids,
                         'org_kg_ids': org_kg_ids}
                
            results[idx] = items
        

    
                
    # metric
    
    
    # save
    torch.save(results, os.path.join(training_args.output_dir, 'train_dataset.pt'))
    logger.info("Output results %s", training_args.output_dir)
                
                
    # if training_args.do_eval and data_args.eval_data_file:
    #     eval_dataloader = DataLoader(eval_dataset, 
    #                                  sampler=SequentialSampler(eval_dataset),
    #                                  batch_size=training_args.per_device_eval_batch_size,
    #                                  collate_fn=data_collator,
    #                                  drop_last=training_args.dataloader_drop_last,
    #                                  num_workers=training_args.dataloader_num_workers,
    #                                  pin_memory=True
    #                                  )
        
    
if __name__ == "__main__":
    main()