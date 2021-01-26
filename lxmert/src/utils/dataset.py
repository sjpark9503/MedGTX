import random
import json
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

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer

from .parameters import DataTrainingArguments

logger = logging.getLogger(__name__)

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
    lang_attention_mask: Optional[List[int]] = None
    kg_attention_mask: Optional[List[int]] = None
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
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, token_type_vocab: dict = None):
        assert os.path.isdir(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.token_type_vocab = token_type_vocab
        self.tokenizer = tokenizer
        self.features = list()
        # if not os.path.isfile(os.path.join(file_path,'cached_feature')):
        logger.info("Creating features from dataset file at %s", file_path)
        # Loading preprocessed data
        self.batch_encoding = torch.load(os.path.join(file_path,'db'))
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
        #     logger.info("Saving features...")
        #     torch.save(self.features,os.path.join(file_path,'cached_feature'))
        # else:
        #     logger.info("Loading features from dataset file at %s", file_path)
        #     self.features = torch.load(os.path.join(file_path,'cached_feature'))

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
        for idx in tqdm(range(len(self.batch_encoding['input']))):
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
        return HeadOnlyDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, token_type_vocab=token_type_vocab)

    if evaluate:
        return _dataset(args.eval_data_file)
    if test:
        return _dataset(args.test_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)

