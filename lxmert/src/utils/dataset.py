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
#from transformers.utils import logging

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
    token_type_ids: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

# class kg_vocab:
#     def __init__(self,path):

class HeadOnlyDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, kg_pad: int):
        assert os.path.isdir(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        if not os.path.isfile(os.path.join(file_path,'cached_feature')):
            logger.info("Creating features from dataset file at %s", file_path)
            # Loading preprocessed data
            lines = torch.load(os.path.join(file_path,'note'))
            self.text_batch_encoding = tokenizer(lines, add_special_tokens=True,padding='max_length', truncation=True, max_length=block_size)
            self.kg_batch_encoding = torch.load(os.path.join(file_path,'kg_norel'))
            self.features = list()
            self.batch2feature(kg_pad)
            logger.info("Saving features...")
            torch.save(self.features,os.path.join(file_path,'cached_feature'))
        else:
            logger.info("Loading features from dataset file at %s", file_path)
            self.features = torch.load(os.path.join(file_path,'cached_feature'))

    def batch2feature(self,kg_pad):
        for idx in tqdm(range(len(self.kg_batch_encoding))):
            inputs = dict([('lang_'+k,self.text_batch_encoding[k][idx]) if 'token_type' not in k else (k, self.text_batch_encoding[k][idx]) for k in self.text_batch_encoding])
            # !!!! ONLY FOR DEBUGGING PURPOSE !!!!
            # temp = np.array(self.kg_batch_encoding[idx])
            # temp[temp>50]=0
            # inputs['kg_input_ids'] = temp.astype(np.int64).tolist()
            inputs['kg_input_ids'] = self.kg_batch_encoding[idx]
            inputs['kg_attention_mask'] = (np.array(inputs['kg_input_ids'])!=kg_pad).astype(np.int64).tolist()
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
    kg_pad: int = 0,
):
    def _dataset(file_path):
        return HeadOnlyDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, kg_pad=kg_pad)

    if evaluate:
        return _dataset(args.eval_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)
