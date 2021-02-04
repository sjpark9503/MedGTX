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
        pass
        # raise ValueError(
        #     f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        # )
    os.makedirs(training_args.output_dir, exist_ok=True)

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
    config.use_ce_pooler = True
    config.pretrained_kg_embedding=""
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
        if 'generation' in training_args.task:
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

    
    # Get data collator
    if 'generation' in training_args.task:
        from utils.data_collator import UniLM_DataCollator
        data_collator = UniLM_DataCollator(tokenizer=tokenizer,
                                           kg_special_token_ids=config.kg_special_token_ids,
                                           prediction=True)
    else:
        raise NotImplementedError("Not implemented task")
    
    def get_dataloader(args, dataset, data_collator):
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


    # *** Define Evaluate function ***
    def evaluate_for_generation(model,
                                tokenizer,
                                dataset,
                                data_loader,
                                training_args,
                                decode_option,
                                mode,
                                ):
        assert mode in ['eval', 'test']
        
        save_file_suffix = '_'.join([str(v) for v in decode_option.values()])
        save_file_path = os.path.join(training_args.output_dir, f"{mode}_outputs_{save_file_suffix}.pt")
        
        if os.path.isfile(save_file_path):
            logger.info(f"You've already had this file, {save_file_path}")
            logger.info(f"Directly load from {save_file_path}")
            results = torch.load(save_file_path)
        else:
            logger.info(f"There is no such file, {save_file_path}")
            logger.info(f"You have to decode...")
            results = {'prd_text': [],
                       'gt_text': [],
                       'gt_graph': [],
                       'ptb_graph': [],
                       'metric': {'bleu': [], 'rouge': [], 'ppl': []},
                       }
            
        '''
        Helper function for evaluation
        ''' 
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
            
        device = training_args.device
        model.to(device)
        model.eval()
            
        # decode
        if len(results['gt_text']) == 0:
            logger.info("start decoding...")
            with torch.no_grad():
                for idx, inputs in tqdm(enumerate(data_loader), total=len(data_loader), desc='Step'):
                    inputs = _prepare_inputs(inputs, device)
                    outputs = model(**inputs, **decode_option)
                    
                    results['prd_text'] += outputs[0]
                    results['gt_graph'] += list(outputs[1])
                    results['gt_text'] += list(outputs[2])
                    if len(outputs) == 4:
                        results['ptb_graph'] += list(outputs[3])
            
        # compute metric: BLEU                
        if len(results['metric']['bleu']) == 0:        
            logger.info("start computing BLEU...")
            from utils.metrics import bleu_all
            
            K = decode_option['search_beam_size']
            for idx in tqdm(range(len(dataset))):
                if K == 1:
                    ref = tokenizer.convert_ids_to_tokens(results['prd_text'][idx], skip_special_tokens=True) # generated text
                    hyp = tokenizer.convert_ids_to_tokens(results['gt_text'][idx], skip_special_tokens=True) # ground truth text
                    results['metric']['bleu'] += bleu_all([ref], hyp)
                else:
                    refs = [tokenizer.convert_ids_to_tokens(outputs[0][idx][k], skip_special_tokens=True) for k in range(K)] # generated text
                    hyp = tokenizer.convert_ids_to_tokens(outputs[2][idx], skip_special_tokens=True) # ground truth text
                    results['metric']['bleu'] += bleu_all(refs, hyp)
        
        # compute metric: PPL
        if isinstance(results['metric']['ppl'], float):
            results['metric']['ppl'] = [results['metric']['ppl']]
        if len(results['metric']['ppl']) == 0:
            logger.info("start computing PPL...")
            final_ppl = 0.0
            with torch.no_grad():
                for idx, inputs in tqdm(enumerate(data_loader), total=len(data_loader), desc='Step'):
                    inputs = _prepare_inputs(inputs, device)
                    batch_mean_ppl = model.decode_for_ppl(**inputs, **decode_option) # batch_mean_ppl
                    final_ppl += batch_mean_ppl
            results['metric']['ppl'] = [final_ppl/len(data_loader)]
            
        # save file
        logger.info("start saving the file...")
        os.makedirs(training_args.output_dir, exist_ok=True)
        for k,v in results.items():
            results[k] = _prepare_outputs(outputs=v)
        torch.save(results, save_file_path)
        
        assert list(results.keys()) == ['prd_text', 'gt_text', 'gt_graph', 'ptb_graph', 'metric']
        assert results['metric']['ppl'][0] > 0 # must exists ppl
        
        return results
    

    def summarize_bleu_score(results, return_results=False):
        assert list(results.keys()) == ['prd_text', 'gt_text', 'gt_graph', 'ptb_graph', 'metric']
        assert list(results['metric']['bleu'][0].keys()) == ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu-a', 'bleu-s']
        
        dataset_size = len(results['gt_text'])
        _keys = results['metric']['bleu'][0].keys()
        bleu_scores = {k:0.0 for k in _keys}
        
        for idx in tqdm(range(dataset_size)):
            for k in _keys:
                bleu_scores[k] += results['metric']['bleu'][idx][k]
                
        for k in _keys:
            bleu_scores[k] /= dataset_size
            print(f"{k}: {bleu_scores[k]:.4f}")

        return bleu_scores if return_results else None
        

    def summarize_ppl(results, return_results=False):
        logger.info('PPL should be computed within the same batch size')
        final_ppl = results['metric']['ppl'][0]
        
        print(f"PPL: {final_ppl:.4f}")

        return final_ppl if return_results else None
    

    def compute_and_summarize_refer_ratio(results, tokenizer, id2node, db_words_pool, num_kg_relations, return_results=False):
        ''' how many refer kg's info? '''
         
        # define pattern for finding px
        import re
        PATTERN = r'(\d*[\d]\. +[a-z][^0-9]+)'
        
        # define refer_ratio metrics
        txt_in_kg_ratio = []
        gen_in_kg_ratio = []
        gen_in_kg_nin_txt_ratio = []
        
        dataset_size = len(results['gt_text'])    
        for idx in tqdm(range(dataset_size)):
            txt = results['gt_text'][idx]
            gen = results['prd_text'][idx]
            kg = results['gt_graph'][idx]
            
            txt = tokenizer.decode(txt, skip_special_tokens=True)
            gen = tokenizer.decode(gen, skip_special_tokens=True)
            
            # extract keywords in text
            _tmp = re.findall(PATTERN, txt)
            txt_keywords = [' '.join(t.split()[1:2]) for t in _tmp]
            _tmp = re.findall(PATTERN, gen)
            gen_keywords = [' '.join(g.split()[1:2]) for g in _tmp]
            
            # extract keywords in graph
            _tmp = [id2node[node_id.item()].replace('\"','') for node_id in kg if node_id not in range(num_kg_relations)]
            kg_keywords = list(set([node_label for node_label in _tmp if node_label in db_words_pool]))
    
            # compare and compute metric 
            txt_keywords_in_kg = set([t for t in txt_keywords if t in ' '.join(kg_keywords).split()])
            gen_keywords_in_kg = set([g for g in gen_keywords if g in ' '.join(kg_keywords).split()])
            gen_keywords_in_kg_nin_txt = gen_keywords_in_kg - txt_keywords_in_kg
            
            txt_in_kg_ratio += [100*len(txt_keywords_in_kg)/len(kg_keywords)]
            gen_in_kg_ratio += [100*len(gen_keywords_in_kg)/len(kg_keywords)]
            gen_in_kg_nin_txt_ratio += [100*len(gen_keywords_in_kg_nin_txt)/len(kg_keywords)]
            
        print('txt_in_kg_ratio', sum(txt_in_kg_ratio) / dataset_size)
        print('gen_in_kg_ratio', sum(gen_in_kg_ratio) / dataset_size)
        print('gen_in_kg_nin_txt_ratio', sum(gen_in_kg_nin_txt_ratio) / dataset_size)
        
        return (txt_in_kg_ratio, gen_in_kg_ratio, gen_in_kg_nin_txt_ratio) if return_results else None

    
    def graph_label_info(data_file, mimic_dir, mode):
        assert mode in ['eval', 'test']
        data_path = os.path.dirname(data_file)
        
        if 'Unified' in data_path:
            node2id = torch.load(os.path.join(data_path, 'unified_node'))
            id2node = {v:k.split('^^')[0] for k,v in node2id.items()}
        # else: 
        #     ENTITY2ID_PATH = f'/home/ssbae/bae/kg_txt_multimodal/preprocessing/{db_name}/entity2id.txt'
        #     id2node = {
        #         int(line.split('\t')[1]) + len(config.kg_special_token_ids):\
        #             line.split('\t')[0].split('^^')[0] for line in open(ENTITY2ID_PATH).read().splitlines()[1:]
        #             }
        #     R = len(config.kg_special_token_ids)
        #     assert R == 3 # {0:'[PAD]', 1:'[MASK]', 2:'[CLS]'}
        
        import pandas as pd
        if '/px' in data_path:
            df_px = pd.read_csv(os.path.join(mimic_dir, 'PRESCRIPTIONS.csv'))
            TOT_DB_WORDS = list(df_px['DRUG'].str.lower().value_counts().index)
            del df_px
        if '/dx,prx' in data_path:
            df_dx = pd.read_csv(os.path.join(mimic_dir, 'D_ICD_DIAGNOSES.csv'))
            df_prx = pd.read_csv(os.path.join(mimic_dir, 'D_ICD_PROCEDURES.csv'))
            TOT_DB_WORDS = list(df_dx['SHORT_TITLE'].str.lower().value_counts().index) \
                + list(df_prx['SHORT_TITLE'].str.lower().value_counts().index)
            del df_dx, df_prx
            
        return {'id2node': id2node, 'db_words_pool': TOT_DB_WORDS}
        
        
    '''
    Main function
    '''
    # 3090 path
    MIMIC_TB_PATH = '/home/sjpark/experiments/kg_txt_multimodal/preprocessing/mimic_table/' 
    # GCP path
    # MIMIC_TB_PATH = ''
        
    # if training_args.do_eval and data_args.eval_data_file:
    #     logger.info("do evaluation for valid dataset")
    #     # get dataset and data loader
    #     eval_dataset = get_dataset(data_args,
    #                                tokenizer=tokenizer,
    #                                token_type_vocab=config.token_type_vocab,
    #                                evaluate=True)
    #     eval_dataloader = get_dataloader(args=training_args,
    #                                      dataset=eval_dataset,
    #                                      data_collator=data_collator)
        
    #     # evaluate
    #     eval_outputs = evaluate_for_generation(model=model,
    #                                            tokenizer=tokenizer,
    #                                            dataset=eval_dataset,
    #                                            data_loader=eval_dataloader,
    #                                            training_args=training_args,
    #                                            decode_option=decode_option,
    #                                            mode='eval')
        
    #     # summarize metrics
    #     _ = summarize_bleu_score(results=eval_outputs, return_results=False)
    #     _ = summarize_ppl(results=eval_outputs, return_results=False)
        
    #     # summarize metrics (for now, px)
    #     if '/px' in data_args.eval_data_file:
    #         infos = graph_label_info(data_file=data_args.eval_data_file, mimic_dir=MIMIC_TB_PATH, mode='eval')
    #         _ = compute_and_summarize_refer_ratio(results=eval_outputs,
    #                                             tokenizer=tokenizer,
    #                                             id2node=infos['id2node'],
    #                                             db_words_pool=infos['db_words_pool'],
    #                                             num_kg_relations=config.num_relations,
    #                                             return_results=False)
            
        
    if training_args.do_eval and data_args.test_data_file:
        logger.info("do evaluation for test dataset")
        # get dataset and data loader
        test_dataset = get_dataset(data_args,
                                   tokenizer=tokenizer,
                                   token_type_vocab=config.token_type_vocab,
                                   test=True)
        test_dataloader = get_dataloader(args=training_args,
                                         dataset=test_dataset,
                                         data_collator=data_collator)
        
        # evaluate
        test_outputs = evaluate_for_generation(model=model,
                                               tokenizer=tokenizer,
                                               dataset=test_dataset,
                                               data_loader=test_dataloader,
                                               training_args=training_args,
                                               decode_option=decode_option,
                                               mode='test')
        
        # summarize metrics
        _ = summarize_bleu_score(results=test_outputs, return_results=False)
        _ = summarize_ppl(results=test_outputs, return_results=False)
        
        # summarize metrics (for now, px)
        if '/px' in data_args.test_data_file:
            infos = graph_label_info(data_file=data_args.test_data_file, mimic_dir=MIMIC_TB_PATH, mode='test')
            _ = compute_and_summarize_refer_ratio(results=test_outputs,
                                                tokenizer=tokenizer,
                                                id2node=infos['id2node'],
                                                db_words_pool=infos['db_words_pool'],
                                                num_kg_relations=config.num_relations,
                                                return_results=False)
                      
    
        
            
        
    

if __name__ == "__main__":
    main()