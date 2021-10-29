import random
import json
import gc
import os
import logging
import pickle
import time
import torch
from filelock import FileLock
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import numpy as np
from itertools import chain

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer

from .parameters import DataTrainingArguments

from utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

"""
Define Dataset & Load
"""
@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    lang_input_ids: List[int]
    kg_input_ids: List[int]
    kg_ext_input_ids: Optional[List[List[int]]] = None
    kg_ext_sum_input_ids: Optional[List[List[int]]] = None
    kg_langinit_input_ids: Optional[List[List[int]]] = None
    lang_attention_mask: Optional[List[int]] = None
    kg_attention_mask: Optional[List[int]] = None
    kg_ext_attention_mask: Optional[List[List[int]]] = None
    kg_ext_sum_attention_mask: Optional[List[List[int]]] = None
    kg_langinit_attention_mask: Optional[List[List[int]]] = None
    kg_label_mask: Optional[List[int]] = None
    kg_label: Optional[List[int]] = None
    label: Optional[List[int]] = None
    rc_indeces: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

class HeadOnlyDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, token_type_vocab: dict = None, knowmix: str = "", gcn: bool = None, task: int=None,):
        assert os.path.isdir(file_path), f"Input file path {file_path} not found"
        self.token_type_vocab = token_type_vocab
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.task = task
        self.knowmix = knowmix
        self.features = list()
        notifier.warning("Creating features from dataset file at %s", file_path)
        # Loading preprocessed data
        self.batch_encoding = torch.load(os.path.join(file_path,'db'))
        if not gcn:
            self.batch_encoding.pop("mask")
            notifier.critical("Turn off GAT")
        else:
            notifier.critical("Turn on GAT")
        if not knowmix:
            self.batch_encoding.pop("knowledge")
            notifier.critical("Turn off fusion & initialization")
        else:
            if "init" in knowmix:
                notifier.critical("Turn on word initialization of KG embedding")
            if "summary" in knowmix:
                notifier.critical("Turn on the admission level fusion")
            if "abs" in knowmix:
                notifier.critical("Turn on the abstract level fusion")
        # # Only for debug
        # self.batch_encoding  = {k:v[:10] for k,v in self.batch_encoding.items()}
        # # Only for debug
        self.batch_encoding['lang'] = dict()
        for text in tqdm(self.batch_encoding['text']):
            sections, notes = list(text.keys()), list(text.values())
            sample = tokenizer(f' {tokenizer.sep_token} '.join([x.strip() for x in notes]) , add_special_tokens=True, padding='max_length', truncation=True, max_length=block_size, return_token_type_ids=False)
            if token_type_vocab:
                sample['token_type_ids'] = self.generate_type_ids(sections, sample['input_ids'])
            for k, v in sample.items():
                if k not in self.batch_encoding['lang']:
                    self.batch_encoding['lang'][k] = list()
                self.batch_encoding['lang'][k].append(v)

        self.batch2feature()

        del self.batch_encoding
        gc.collect()

    def generate_type_ids(self, sections, tokens):
        idx = 0
        type_ids = list()
        for token in tokens:
            type_ids.append(self.token_type_vocab[sections[idx]])
            if token == self.tokenizer.sep_token_id:
                idx += 1
                if idx >= len(sections):
                    idx = len(sections)-1
        return type_ids

    def batch2feature(self): 
        # Set External Token Length
        if 'knowledge' in self.batch_encoding:
            if 'px_' in self.file_path:
                ext_max_len = 2048
            elif 'dx,prx_' in self.file_path:
                ext_max_len = 768
            else:
                raise ValueError ("Cannot find DB type in file path")
        for idx in tqdm(range(len(self.batch_encoding['input']))):
        # for idx in tqdm(range(128)):
            inputs = dict([('lang_'+k,self.batch_encoding['lang'][k][idx]) if 'token_type' not in k else (k, self.batch_encoding['lang'][k][idx]) for k in self.batch_encoding['lang']])
            inputs['kg_input_ids'] = self.batch_encoding['input'][idx]
            if 'mask' in self.batch_encoding:
                inputs['kg_attention_mask'] = self.batch_encoding['mask'][idx]
            if 'label' in self.batch_encoding:
                if 'label_mask' in self.batch_encoding:
                    inputs['kg_label'] = self.batch_encoding['label'][idx]
                    inputs['kg_label_mask'] = self.batch_encoding['label_mask'][idx]
                else:
                    inputs['label'] = self.batch_encoding['label'][idx]
            if 'rc_index' in self.batch_encoding:
                inputs['rc_indeces'] = self.batch_encoding['rc_index'][idx]
            if 'knowledge' in self.batch_encoding:
                if "abs" in self.knowmix:
                    abs_indices = [x for _idx, x in enumerate(self.batch_encoding['knowledge'][idx]) if (_idx>1) and not x]
                    processed_knowledge = list()
                    for node_idx in range(len(self.batch_encoding['knowledge'][idx])):
                        if node_idx in abs_indices:
                            processed_knowledge.append(" ".join([_s for _idx, _mask, _s in enumerate(zip(inputs['kg_attention_mask'][node_idx],self.batch_encoding['knowledge'][idx])) if (_idx>1) and (_mask!=0)]).strip())
                        else:
                            processed_knowledge.append("")
                    tokenized_knowledge = self.tokenizer(processed_knowledge, add_special_tokens=False, padding='max_length', max_length=64, return_token_type_ids=False)
                    inputs['kg_ext_input_ids'] = tokenized_knowledge['input_ids']
                    inputs['kg_ext_attention_mask'] = tokenized_knowledge['attention_mask']
                if "init" in self.knowmix:
                    if "linearize" in self.knowmix:
                        tokenized_init = self.tokenizer((" ".join(self.batch_encoding['knowledge'][idx])).strip(), add_special_tokens=False, padding='max_length', max_length=ext_max_len, return_token_type_ids=False)
                        inputs['kg_input_ids'] = tokenized_init['input_ids']
                        inputs['kg_attention_mask'] = tokenized_init['attention_mask']
                        inputs.pop('rc_indeces')
                        inputs.pop('kg_label')
                        inputs.pop('kg_label_mask')
                    else:
                        tokenized_init = self.tokenizer(self.batch_encoding['knowledge'][idx], add_special_tokens=False, padding='max_length', max_length=64, return_token_type_ids=False)
                        inputs['kg_langinit_input_ids'] = tokenized_init['input_ids']
                        inputs['kg_langinit_attention_mask'] = tokenized_init['attention_mask']
                if "summary" in self.knowmix:
                    summarized_knowledge = self.tokenizer((" ".join(self.batch_encoding['knowledge'][idx])).strip(), add_special_tokens=False, padding='max_length', max_length=ext_max_len, return_token_type_ids=False)
                    inputs['kg_ext_sum_input_ids'] = summarized_knowledge['input_ids']
                    inputs['kg_ext_sum_attention_mask'] = summarized_knowledge['attention_mask']

            feature = InputFeatures(**inputs)
            self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    test: bool = False,
    token_type_vocab: dict = None
):
    def _dataset(file_path):
        return HeadOnlyDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, token_type_vocab=token_type_vocab, knowmix=args.knowmix, gcn=args.gcn, task = args.task)

    if evaluate:
        return _dataset(args.eval_data_file)
    elif test:
        return _dataset(args.test_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)

