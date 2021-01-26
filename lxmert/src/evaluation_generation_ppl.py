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
from model import LxmertForRanking
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
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # convert remaining args to dict (decode_option)
    for r_arg in remaining_args:
        if r_arg.find('decode_option') != -1:
            from ast import literal_eval
            decode_option = literal_eval(r_arg.split('=')[1])
        else:
            raise ValueError("You have to add decode_option")
        decode_option['given_lang_tokens'] = int(decode_option['given_lang_tokens'])
        decode_option['search_beam_size'] = int(decode_option['search_beam_size'])
        assert set(decode_option.keys()) == set(['perturb_type', 'given_lang_tokens', 'clean_outputs', 'given_gt_length', 'search_beam_size'])
        
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
        if training_args.task in ['generation']:
            from model import LxmertForGeneration
            model = LxmertForGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError("Not implemented task: %s", training_args.task)
    else:
        # logger.info("Training new model from scratch")
        # model = LxmertForRanking(config)
        raise NotImplementedError("There is no model_name_or_path")

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
    if training_args.do_eval and data_args.eval_data_file:
        eval_dataset = get_dataset(data_args,
                                   tokenizer=tokenizer,
                                   token_type_vocab=config.token_type_vocab,
                                   evaluate=True
                                   )
        # test_dataset = get_dataset(data_args,
        #                            tokenizer=tokenizer,
        #                            token_type_vocab=config.token_type_vocab,
        #                            test=True
        #                            ) if training_args.do_eval else None
    
    # Get data collator
    if training_args.task == 'generation':
        from utils.data_collator import UniLM_DataCollator
        data_collator = UniLM_DataCollator(tokenizer=tokenizer,
                                           kg_special_token_ids=config.kg_special_token_ids,
                                           prediction=True)
    else:
        raise NotImplementedError("Not implemented task")
    
    
    # Helper function for evaluation
    def _get_dataloader(args, dataset, data_collator):
        sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator, 
            drop_last=args.dataloader_drop_last,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )
            
    def _prepare_inputs(inputs, device):
        if isinstance(inputs, dict):
            for k,v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
        return inputs
            
    def _prepare_outputs(outputs):
        if isinstance(outputs, list):
            for i, o in enumerate(outputs):
                if isinstance(o, torch.Tensor):
                    outputs[i] = o.cpu()
        return outputs
    
    if training_args.do_eval and data_args.eval_data_file:
        eval_dataloader = _get_dataloader(args=training_args, dataset=eval_dataset, data_collator=data_collator)
        # test_dataloader = _get_dataloader(args=training_args, dataset=test_dataset, data_collator=data_collator)
    
    
    device = training_args.device
    model.to(device)
    model.eval()
    
    eval_outputs, test_outputs = None, None
    if training_args.do_eval and data_args.eval_data_file:        
        eval_outputs = {'prd_text': [],
                        'gt_text': [],
                        'gt_graph': [],
                        'ptb_graph': [],
                        'metric': {'bleu': [], 'rouge': [], 'ppl': []},
        }
        eval_dataset_size = len(eval_dataset)
        
        # 1. generate text(=decoding)
        logger.info("start decoding...")
        final_ppl = 0.0
        with torch.no_grad():
            for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='Step'):
                inputs = _prepare_inputs(inputs, device)
                batch_ppl = model.decode_for_ppl(**inputs, **decode_option) # batch_mean_ppl
                final_ppl += batch_ppl                
            print(f'PPL: {final_ppl/len(eval_dataloader)}')
                    
        # # 2. evaluate metrics
        # logger.info("start evaluation...")
        
        # from utils.metrics import bleu_all
        # K = decode_option['search_beam_size']
        # for idx in tqdm(range(eval_dataset_size)):
        #     if K == 1:
        #         ref = tokenizer.convert_ids_to_tokens(eval_outputs['prd_text'][idx], skip_special_tokens=True) # generated text
        #         hyp = tokenizer.convert_ids_to_tokens(eval_outputs['gt_text'][idx], skip_special_tokens=True) # ground truth text
        #         eval_outputs['metric']['bleu'] += bleu_all([ref], hyp)
        #     else:
        #         refs = [tokenizer.convert_ids_to_tokens(outputs[0][idx][k], skip_special_tokens=True) for k in range(K)] # generated text
        #         hyp = tokenizer.convert_ids_to_tokens(outputs[2][idx], skip_special_tokens=True) # ground truth text
        #         eval_outputs['metric']['bleu'] += bleu_all(refs, hyp)
    
        # # 3. Summarize metrics
        # _keys = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-a', 'bleu-s']
        # bleu_scores = {k:0.0 for k in _keys}
        # for idx in tqdm(range(eval_dataset_size)):            
        #     for k in _keys:
        #         bleu_scores[k] += eval_outputs['metric']['bleu'][idx][k]
                
        # print('bleu-1: ', bleu_scores['bleu-1']/eval_dataset_size)
        # print('bleu-2: ', bleu_scores['bleu-2']/eval_dataset_size)
        # print('bleu-3: ', bleu_scores['bleu-3']/eval_dataset_size)
        # print('bleu-4: ', bleu_scores['bleu-4']/eval_dataset_size)
        # print('bleu-a: ', bleu_scores['bleu-a']/eval_dataset_size)
        # print('bleu-s: ', bleu_scores['bleu-s']/eval_dataset_size)
        
    # # 3. Save outputs
    # if training_args.do_eval and data_args.eval_data_file:
    #     os.makedirs(training_args.output_dir, exist_ok=True)
    #     for k,v in eval_outputs.items():
    #         eval_outputs[k] = _prepare_outputs(outputs=v)
        
    #     file_suffix = '_'.join([str(v) for v in decode_option.values()])    
    #     torch.save(eval_outputs, os.path.join(training_args.output_dir, f"eval_outputs_{file_suffix}.pt"))
    
    
    

if __name__ == "__main__":
    main()