import os
from tqdm import tqdm
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import torch
from transformers import AutoModel, AutoTokenizer

# bluebert models
BlueBERT_MODELCARD = [
    'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16',
    'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
    'bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16'
]

# googlebert models
GoogleBERT_MODELCARD = [
    'google/bert_uncased_L-2_H-128_A-2', 
    'google/bert_uncased_L-4_H-128_A-2', 
    'google/bert_uncased_L-6_H-128_A-2', 
    'google/bert_uncased_L-2_H-512_A-2', 
    'google/bert_uncased_L-4_H-512_A-2', 
    'google/bert_uncased_L-6_H-512_A-2',
]

@dataclass
class EhrKgNode2IdMapping:
    '''
    This class could be only implemented,
    as the form of "entity2id.txt" (or "node2id.txt" in the feature)
    '''
    exp_path: str
    file_name: str = field(default='entity2id.txt') # actually it means node2id.txt (they all have entities and literals)
    kg_special_token_ids: dict = field(default_factory=lambda: {"PAD":0,"MASK":1})
    skip_first_line: bool = True

    def get_lines(self):
        file_path = os.path.join(self.exp_path, self.file_name)
        with open(file_path) as f:
            lines = f.read().splitlines()
            if self.skip_first_line:
                lines = lines[1:]
        return lines

    def get_id2literal(self) -> dict:
        lines = self.get_lines()
        lines_literal = list(filter(None, [self._get_literal(line) for line in lines]))
        id2literal = {self._make_id2key(line) : self._make_str2val(line) for line in lines_literal}
        return id2literal

    def get_id2entity(self) -> dict:
        ''' actually means (entity => node)'''
        lines = self.get_lines()
        id2entity = {self._make_id2key(line) : self._make_str2val(line) for line in lines}
        return id2entity

    def _get_literal(self, line: str) -> str:
        (node, node_id) = line.split('\t')
        _check_node = node.split('^^')
        if len(_check_node) == 2:
            literal = _check_node[0].replace("\"","") # clean "
            return literal + '\t' + node_id

    def _make_id2key(self, line: str) -> int:
        _id = int(line.split('\t')[1])
        _add = len(self.kg_special_token_ids) # len(config.kg_special_token_ids)
        key = (_id + _add)
        return key

    def _make_str2val(self, line: str) -> str:
        val = line.split('\t')[0].split('^^')[0]
        return val


_no_default = object()
@dataclass
class EhrKgNode2EmbeddingMapping(EhrKgNode2IdMapping):
    
    model_name_or_path: str = _no_default
    # kg_special_token_ids: dict = field(default_factory={"PAD":0,"MASK":1})
    # tokenizer_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # )

    def __post_init__(self):
        if self.model_name_or_path is _no_default:
            raise TypeError("__init__ missing 1 required argument: 'model_name_or_path'")

    def _load_model_and_tokenizer(self):
        # load model
        if self.model_name_or_path:
            model = AutoModel.from_pretrained(self.model_name_or_path)
        else:
            raise ValueError("There is no (pre-trained) model name or path.")
        # load tokenizer
        if self.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        else:
            raise ValueError("There is no (pre-trained) tokenizer name or path.")
        return model, tokenizer

    def get_literal_embeddings_from_model(self):
        model, tokenizer = self._load_model_and_tokenizer() # load (pre-trained) model and tokenizer
        id2literal = self.get_id2literal() # get mapping dict
        
        def _convert_to_model_input(literal: str, tokenizer) -> List[str]:
            return tokenizer(text=literal, return_tensors='pt', padding=True, truncation=True)
        
        id2literalembedding = {}
        for k, v in tqdm(id2literal.items()):
            encoded_input = _convert_to_model_input(literal=v, tokenizer=tokenizer)
            _, output = model(**encoded_input)
            id2literalembedding[k] = output.cpu().detach()
        return id2literalembedding

    def save_literal_embeddings_from_model(self, save_file_dir: str, save_file_name: str = 'id2literalembedding.pt'):
        if not os.path.isdir(save_file_dir):
            os.mkdir(save_file_dir)
        save_file_path = os.path.join(save_file_dir, save_file_name)
        id2literalembedding = self.get_literal_embeddings_from_model()
        torch.save(id2literalembedding, save_file_path)