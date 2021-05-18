import random
import json
import os
import pickle
import time
import torch
from filelock import FileLock
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)
InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

@dataclass
class NodeClassification_DataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    kg_special_token_ids: dict
    kg_size: int
    align: bool = False
    n_negatives: int = 1
    edge_cls: bool = False
    mlm: bool = True
    mlm_probability: float = 0.15
    contrastive: bool = False
    prediction: bool = False

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        batch = self._tensorize_batch(features)

        if not self.prediction:
            # Construct batch for Masked LM
            masked_texts, lm_label = self.mask_tokens(batch['lang_input_ids'])
            # Construct batch for Masekd LP
            if not self.contrastive:
                masked_subs, kg_label_mask, kg_padding_mask = self.mask_kg(batch['kg_input_ids'], batch['kg_label_mask'])
                batch['kg_input_ids'] = masked_subs
                batch['kg_label_mask'] = kg_label_mask
                batch['kg_padding_mask'] = kg_padding_mask
            else:
                batch['kg_label_mask'] = None
            batch['lang_input_ids'] = masked_texts
            batch['lm_label'] = lm_label
            # Construct batch for Alignment loss
            if self.align:
                batch = self.negative_sampling(batch, len(features))
        else:
            batch['lang_input_ids'] = batch['lang_input_ids']
            batch['lm_label'] = None
            batch['kg_input_ids'] = batch['kg_input_ids']
            batch['kg_label'] = None

        return batch

    def _tensorize_batch(self,features: List[Dict]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        batch = {}

        for k, v in first.items():
            if 'rc' in k:
                if self.edge_cls:
                    batch[k] = [f[k] for f in features]
                continue
            if v is not None:
                if (k == "kg_attention_mask") and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        if (len(v.shape) == 3):
                            batch[k] = torch.stack([f[k] for f in features]).permute(0,3,1,2)
                        else:
                            batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                elif not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

        return batch

    def negative_sampling(self,batch, batch_size) -> Dict[str, torch.Tensor]:
        for k, v in batch.items():
            if v is not None:
                if ('rc' in k) or ('label' in k):
                    continue
                elif 'kg' not in k:
                    batch[k] = torch.cat([batch[k].detach().clone()[(torch.arange(batch_size) + idx) % batch_size] for idx in range(self.n_negatives+1)],dim=0)

                else:
                    batch[k] = torch.cat([batch[k].detach().clone() for _ in range(self.n_negatives + 1)],dim=0)

        batch['cross_label'] = torch.cat([torch.ones(batch_size, dtype=torch.long),
                                             torch.zeros(batch_size*self.n_negatives, dtype=torch.long)],dim=0)
        return batch

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mask_kg(self, inputs: torch.Tensor, entity_mask = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.kg_special_token_ids['MASK'] is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        ignore_masking = ~torch.tensor(entity_mask, dtype=torch.bool)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        if entity_mask is not None:
            probability_matrix.masked_fill_(ignore_masking, value=0.0)
        padding_mask = ~inputs.eq(self.kg_special_token_ids['PAD'])
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.kg_special_token_ids['MASK']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_nodes = torch.randint(len(self.kg_special_token_ids),self.kg_size, inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_nodes[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, masked_indices, padding_mask.float()

    def kg_negative_sampling(self, inputs: torch.Tensor, entity_mask = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        return NotImplementedError("Not Support Yet")

@dataclass
class NegativeSampling_DataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    kg_special_token_ids: dict
    n_negatives: int = 1
    prediction: bool = False

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        batch = self._tensorize_batch(features)
        batch_size = len(features)

        if not self.prediction:
            # if self.NCE:
            #     shuffled_idx = (torch.arange(batch_size) + torch.randint(1, batch_size - 1, (1, 1)).item()) % batch_size
            #     batch['negative_kg_input_ids'] = batch['kg_input_ids'].detach().clone()[shuffled_idx]
            #     batch['label'] = None
            # else:
            batch['label'] = torch.cat([torch.ones(batch_size, dtype=torch.long),
                                             torch.zeros(batch_size*self.n_negatives, dtype=torch.long)],dim=0)
        else:
            return NotImplementedError("Not Support Yet")

        return batch

    def _tensorize_batch(self,features: List[Dict]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        batch = {}

        for k, v in first.items():
            if ('label' in k) or ('rc' in k):
                continue
            if (v is not None) and (not isinstance(v, str)):
                if (k == "kg_attention_mask"):
                    if isinstance(v, torch.Tensor):
                        if (len(v.shape) == 3):
                            batch[k] = torch.stack([f[k] for f in features]).permute(0,3,1,2)
                        else:
                            batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                else:
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

                    if (k == "kg_input_ids"):
                        batch['kg_padding_mask'] = ~batch[k].detach().clone().eq(self.kg_special_token_ids['PAD'])

                # if not NCE:
                if 'kg' not in k:
                    batch_size = len(features)
                    batch[k] = torch.cat([batch[k].detach().clone()[(torch.arange(batch_size) + idx) % batch_size] for idx in range(self.n_negatives+1)],dim=0)
                else:
                    batch[k] = torch.cat([batch[k].detach().clone() for _ in range(self.n_negatives + 1)],dim=0)
                    if (k == "kg_input_ids"):
                        batch['kg_padding_mask'] = torch.cat([batch['kg_padding_mask'].detach().clone() for _ in range(self.n_negatives + 1)],dim=0)

        return batch

@dataclass
class UniLM_DataCollator:
    """
    Data collator used for seq2seq language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    kg_special_token_ids: dict
    # kg_size: int
    mlm: bool = True
    mlm_probability: float = 0.15
    prediction: bool = False

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        batch = self._tensorize_batch(features)

        if not self.prediction:
            # Text Part
            masked_texts, lm_label = self.mask_tokens_with_sep(batch['lang_input_ids'])
            casual_mask = self.make_lang_attention_mask(batch['lang_attention_mask'])
            batch['lang_input_ids'] = masked_texts
            batch['lm_label'] = lm_label
            batch['lang_attention_mask'] = casual_mask
                    
            # Graph Part
            _, kg_label_mask, kg_padding_mask = self.make_kg_padding_mask(batch['kg_input_ids']) # only need padding_mask
            batch['kg_padding_mask'] = kg_padding_mask
            batch['kg_label_mask'] = kg_label_mask
            batch['kg_label'] = None
            
            # check padding by using assert
            # self._assert_padding(batch)
        else:
            '''
            for decoding (generation)
            '''
            batch['lang_input_ids'] = batch['lang_input_ids']
            batch['lm_label'] = None
            batch['kg_input_ids'] = batch['kg_input_ids']
            batch['kg_label'] = None
            
            casual_mask = self.make_lang_attention_mask(batch['lang_attention_mask'])
            batch['lang_attention_mask'] = casual_mask
            _, kg_label_mask, kg_padding_mask = self.make_kg_padding_mask(batch['kg_input_ids']) # only need padding_mask
            batch['kg_padding_mask'] = kg_padding_mask

        return batch

    def _tensorize_batch(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        batch = {}

        for k, v in first.items():
            if 'rc' in k:
                # we don't need to use graph part
                continue
            if v is not None:
                if (k == "kg_attention_mask") and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        if (len(v.shape) == 3):
                            batch[k] = torch.stack([f[k] for f in features]).permute(0,3,1,2)
                        else:
                            batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                elif not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

        return batch

    def mask_tokens_with_sep(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # To learn <eos> part, we replace 50% masked [SEP] token.
        if self.tokenizer._sep_token is not None:
            sep_mask = labels.eq(self.tokenizer.sep_token_id)
            probability_matrix.masked_fill_(sep_mask, value=0.5) 
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def make_kg_padding_mask(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.kg_special_token_ids['MASK'] is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for padding mask in the graph part."
            )
        padding_mask = ~inputs.eq(self.kg_special_token_ids['PAD'])
        masked_indices = torch.zeros(inputs.shape, dtype=torch.float)
        
        return inputs, masked_indices, padding_mask.float()

    def make_lang_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = attention_mask.shape
        seq_ids = torch.arange(seq_length)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = attention_mask.unsqueeze(2) * causal_mask
        return causal_mask

    def _assert_padding(self, batch):
        '''
        assert for grpah part
        a) lengths are the same when we exempt for padding mask
        b) label_mask는 다 0 = mask되는 부분이 없기 때문
        c) attention_mask -> padding check (크게 신경쓰지 않아도 되는 부분)
        '''
        graph_a1 = torch.sum(batch['kg_input_ids'].not_equal(self.kg_special_token_ids['PAD']), axis=1)
        graph_a2 = torch.sum(batch['kg_padding_mask'], axis=1)
        assert graph_a1 == graph_a2

        graph_b = torch.all(batch['kg_label_mask'].eq(self.tokenizer.pad_token_id))
        assert graph_b == True

        '''
        assert for lang part
        a) legnths are the same when we exempt for padding mask
        '''
        text_a1 = torch.sum(batch['lang_input_ids'].not_equal(self.tokenizer.pad_token_id), axis=1)
        if batch['lang_attention_mask'].dim() == 2:
            text_a2 = torch.sum(batch['lang_attention_mask'], axis=1)
        elif batch['lang_attention_mask'].dim() == 3:
            text_a2 = torch.max(torch.sum(batch['lang_attention_mask'], axis=2), axis=1)[0]
        assert text_a1 == text_a2

@dataclass
class AdmLvlPred_DataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    kg_special_token_ids: dict
    num_kg_labels: int
    prediction: bool = False

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        batch = self._tensorize_batch(features)
        batch_size = len(features)

        # else:
        #     return NotImplementedError("Not Support Yet")

        return batch

    def _tensorize_batch(self,features: List[Dict]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        batch = {}

        for k, v in first.items():
            if (v is not None) and (not isinstance(v, str)):
                if (k == 'label'):
                    batch[k] = torch.stack([torch.zeros(self.num_kg_labels).index_fill_(0,torch.LongTensor(f[k]),1) for f in features])
                    continue
                if (k == "kg_attention_mask"):
                    if isinstance(v, torch.Tensor):
                        if (len(v.shape) == 3):
                            batch[k] = torch.stack([f[k] for f in features]).permute(0,3,1,2)
                        else:
                            batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                else:
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

                    if (k == "kg_input_ids"):
                        batch['kg_padding_mask'] = ~batch[k].detach().clone().eq(self.kg_special_token_ids['PAD'])

        return batch

@dataclass
class ErrorDetection_DataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    kg_special_token_ids: dict
    #num_kg_labels: int
    kg_size: int
    label_domain: str
    prediction: bool = False

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        batch = self._tensorize_batch(features)
        if 'graph' == self.label_domain:
            self.corruption_probability=0.1
            batch = self.kg_corruption(batch)
        elif 'text' == self.label_domain:
            self.corruption_probability=0.2
            batch = self.text_corruption(batch)
        else:
            raise ValueError("Task not supported")
        batch.pop('label')

        return batch

    def _tensorize_batch(self,features: List[Dict]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        batch = {}

        for k, v in first.items():
            if ('rc' in k) or ('label_mask' in k):
                continue
            if (v is not None) and not isinstance(v,str):
                if k == "kg_attention_mask":
                    if isinstance(v, torch.Tensor):
                        if (len(v.shape) == 3):
                            batch[k] = torch.stack([f[k] for f in features]).permute(0,3,1,2)
                        else:
                            batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                else:
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

        return batch

    def kg_corruption(self, batch: dict):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        inputs = batch['kg_input_ids']
        labels = batch['label']
        ignore_corruption = labels.eq(-100)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(inputs.shape, self.corruption_probability)
        probability_matrix.masked_fill_(ignore_corruption, value=0.0)
        must_corrupt_idx=torch.max(torch.rand(ignore_corruption.shape)*(~ignore_corruption),dim=1)[-1]
        probability_matrix.scatter_(1,must_corrupt_idx.unsqueeze(1),1.0)
        corruption_indeces = torch.bernoulli(probability_matrix).bool()

        # if 'deletion' in self.task:
        #     inputs[corruption_indeces] = self.kg_special_token_ids['PAD']
        #     deleted_label = torch.zeros(inputs.size(0),self.num_kg_labels)
        #     for idx in corruption_indeces.nonzero():
        #         deleted_label[idx[0],labels[idx[0],idx[1]]]=1
        #         if 'kg_attention_mask' in batch:
        #             batch['kg_attention_mask'][idx[0],:,idx[1]]=0
        #     batch['kg_label']=deleted_label
        # elif 'replacement' in self.task:
        inputs_origin = inputs.clone()
        random_nodes = torch.randint(len(self.kg_special_token_ids),self.kg_size, inputs.shape, dtype=torch.long)
        inputs[corruption_indeces] = random_nodes[corruption_indeces]
        batch['kg_label'] = (~(inputs_origin == inputs)).float()
        # else:
        #     raise ValueError('task not supported ')
        
        batch['kg_input_ids'] = inputs
        batch['kg_padding_mask'] = ~inputs.eq(self.kg_special_token_ids['PAD'])
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return batch

    def text_corruption(self, batch: dict):
        mask_labels = []
        for _input in batch["lang_input_ids"].tolist():
            # ref_tokens = []
            # for id in _input:
            tokens = self.tokenizer.convert_ids_to_tokens(_input)
            ref_tokens = tokens
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = torch.tensor(mask_labels)
        inputs, labels = self.mask_tokens(batch["lang_input_ids"], batch_mask)
        batch['lang_input_ids'] = inputs
        batch['lm_label'] = labels
        batch['kg_padding_mask'] = ~batch['kg_input_ids'].eq(self.kg_special_token_ids['PAD'])
        return batch

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.corruption_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))] 
        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        inputs_origin = inputs.clone()
        probability_matrix = mask_labels.bool()

        masked_indices = probability_matrix.bool()

        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        labels = (~(inputs_origin == inputs)).float()

        return inputs, labels

@dataclass
class Evaluation_DataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    task : str
    kg_special_token_ids: dict

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        batch = self._tensorize_batch(features)
        batch_size = len(features)

        return batch

    def _tensorize_batch(self,features: List[Dict]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        batch = {}

        for k, v in first.items():
            if ('label' in k) or ('rc' in k):
                continue
            if (v is not None) and (not isinstance(v, str)):
                if (k == "kg_attention_mask"):
                    if isinstance(v, torch.Tensor):
                        if (len(v.shape) == 3):
                            batch[k] = torch.stack([f[k] for f in features]).permute(0,3,1,2)
                        else:
                            batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                else:
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

                    if (k == "kg_input_ids"):
                        batch['kg_padding_mask'] = ~batch[k].detach().clone().eq(self.kg_special_token_ids['PAD'])

        return batch