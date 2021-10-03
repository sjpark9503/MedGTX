# Base pkgs
import json
import gc
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import numpy as np
import itertools
from dataclasses import dataclass
from core.model import ProbingEncModel
import copy
import os
# Transformers
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer


from transformers import AutoTokenizer, AutoModel,AutoConfig,AutoModelForMaskedLM
from torch.utils.data import DataLoader
import wandb

"""
Define Dataset & Load
"""
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
    def __init__(self, input_ids, **kwargs):
        self.input_ids = input_ids
        for k, v in kwargs.items():
            self.__setattr__(k,v)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def split_digits(wps):
    # further split numeric wps
    toks = []
    for wp in wps:
        if set(wp).issubset(set('#0123456789')) and set(wp) != {'#'}: # numeric wp - split digits
            for i, dgt in enumerate(list(wp.replace('#', ''))):
                prefix = '##' if (wp.startswith('##') or i > 0) else ''
                toks.append(prefix + dgt)
        else:
            toks.append(wp)
    return toks

class PredDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, method:str, pe_type:int):
        self.tokenizer = tokenizer
        self.method = method
        self.pe_type = pe_type
        self.features = list()
        self.batch_encoding = torch.load(file_path)
        self.batch_encoding['raw_input'] = self.batch_encoding['input']
        self.batch_encoding['input'] = tokenizer(self.batch_encoding['input'], add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True, return_offsets_mapping=True)
        self.target_vocab = list(set(self.batch_encoding['label']))

        self.batch2feature()

        del self.batch_encoding
        gc.collect()

    def batch2feature(self):
        for idx in tqdm(range(len(self.batch_encoding['label']))):
            inputs = {k:self.batch_encoding['input'][k][idx] for k in self.batch_encoding['input'].keys()}
            if "clz" in self.method:
                inputs['labels'] = self.tokenizer(self.batch_encoding['label'][idx], add_special_tokens=False)['input_ids']
            else:
                inputs['labels'] = int(self.batch_encoding['label'][idx])
            if ("number_offset" in self.batch_encoding) and (self.pe_type !=0):
                inputs['additional_pe'] = self._create_add_pe(self.batch_encoding['number_offset'][idx], self.batch_encoding['input']['offset_mapping'][idx])
            inputs['raw_input'] = self.batch_encoding['raw_input'][idx]
            inputs['number_offset'] = self.batch_encoding['number_offset'][idx]
            feature = InputFeatures(**inputs)
            self.features.append(feature)

    def _create_add_pe(self, number_offset, token_offset):
        token_offset.reverse()
        add_pe = list()
        if self.pe_type == 1:
            for pos in token_offset:
                if pos == (0, 0):
                    pe_idx = 0
                    add_pe.insert(0,pe_idx)
                    continue
                FLAG = False
                # check pos is contained in number
                for num_pos in number_offset:
                    if (num_pos[0]<=pos[0]) and (num_pos[1]>=pos[1]):
                        FLAG = True
                        break
                if FLAG:
                    pe_idx+=1
                else:
                    pe_idx=0
                add_pe.insert(0,pe_idx)
        else:
            raise NotImplementedError("Other type of positional encoding is not supported.")

        return add_pe


    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

def get_dataset(
    tokenizer: PreTrainedTokenizer,
    file_path: str,
    method: str,
    pe_type: int):
    dataset = PredDataset(tokenizer=tokenizer, file_path=file_path, method=method,pe_type=pe_type)
    target_vocab = dataset.target_vocab

    return dataset, list(set(target_vocab))

InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

@dataclass
class PredDataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    method: str
    vocab_ignore_idx: Optional[List] = None
    prediction: bool = False

    def _pad(self, encoded_inputs: List[Dict], max_length, padding_side="right"):
        required_input = encoded_inputs["input_ids"]
        needs_to_be_padded = len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if padding_side == "right":
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] += [self.tokenizer.pad_token_type_id] * difference
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] += [1] * difference
                if "additional_pe" in encoded_inputs:
                    encoded_inputs["additional_pe"] += [0] * difference
                encoded_inputs["input_ids"] = required_input + [self.tokenizer.pad_token_id] * difference
                encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference

            elif padding_side == "left":
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.tokenizer.pad_token_type_id] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                if "additional_pe" in encoded_inputs:
                    encoded_inputs["additional_pe"] = [0] * difference + encoded_inputs["additional_pe"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + required_input
                encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)

        return encoded_inputs


    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        max_length = max([len(f.input_ids) for f in features])
        pre_batch = {}
        for f in features:
            encoded_inputs = vars(f)
            outputs = self._pad(
                encoded_inputs,
                max_length,
            )
            for key, value in outputs.items():
                if key not in pre_batch:
                    pre_batch[key] = []
                pre_batch[key].append(value)
        batch = {}
        for k in pre_batch.keys():
            if k in ['number_offset','raw_input','offset_mapping']:
                batch[k] = pre_batch[k]
            elif (k=='labels') and ('clz' in self.method):
                batch[k] = torch.tensor(list(itertools.chain(*pre_batch[k]))).cuda()
            else:
                batch[k] = torch.tensor(pre_batch[k]).cuda()
        if 'clz' in self.method:
            batch['label_mask'] = (batch['input_ids']==self.tokenizer.mask_token_id).cuda()
            batch['vocab_mask'] = torch.zeros(len(self.tokenizer)).index_fill_(dim=0,index=torch.tensor(self.vocab_ignore_idx),value=-1e10).cuda()

        return batch

count = 0
for task_num in [1]:
    for method in ["bin-frz-full"]:
        for model_name in ["ALBERT"]:
            for pe in [1]:
                for notation in ["sci-char"]:#"word","char"]:
                    for uom in ["biomedical"]:
                        task_name = {
                            1: 'comparison', 
                            2: 'min_max', 
                            3: 'sort', 
                            4: 'convert', 
                            5: 'num2word', 
                            6: 'val_range', 
                            7: 'uom_validity'
                        }[task_num]

                        tokenizer_name = {
                            "BERT":'bert-base-uncased',
                            "BioBERT":'dmis-lab/biobert-v1.1',
                            "BlueBERT":'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
                            "ALBERT":'albert-base-v2',
                            "RoBERTa":'roberta-base',
                            "GPT-3":'EleutherAI/gpt-neo-2.7B',
                            "UniLM":'microsoft/unilm-base-cased',
                            'Rand':'bert-base-uncased'
                        }[model_name]

                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                        num_added_tokens =  tokenizer.add_special_tokens({'additional_special_tokens':["[DEC]", "[EXP]", "[UNK_UNIT]", "[UNK_PFX]"]})

                        # Define model class
                        MODEL_CLASS = {
                            "clz": AutoModelForMaskedLM,
                            "bin": AutoModel
                        }

                        # Load Config
                        for seed in [42,1234,123]:
                            if os.path.isfile(f"/pretrained_models/{task_name}/{method}/{model_name}-pe{pe}-{notation}/{uom}/RNG{seed}/config.json") and os.path.isfile(f"/pretrained_models/{task_name}/{method}/{model_name}-pe{pe}-{notation}/{uom}/RNG{seed}/pytorch_model.bin"):
                                selected_seed = seed
                                break
                            selected_seed = None
                        if selected_seed is None:
                            continue
                        config = AutoConfig.from_pretrained(f"/pretrained_models/{task_name}/{method}/{model_name}-pe{pe}-{notation}/{uom}/RNG{seed}/config.json")

                        # Load Model
                        model = ProbingEncModel(config=config, return_dict=True, tokenizer=tokenizer)
                        enc = MODEL_CLASS[method.split("-")[0]].from_pretrained(tokenizer_name)
                        enc.pe_type = pe
                        enc.resize_token_embeddings(len(tokenizer))
                        model.encoder = enc
                        org_sdict = copy.deepcopy(model.state_dict())
                        model.load_state_dict(torch.load(f"/pretrained_models/{task_name}/{method}/{model_name}-pe{pe}-{notation}/{uom}/RNG{seed}/pytorch_model.bin"))

                        # Get loaded state dict
                        new_sdict = model.state_dict()

                        # Double-check the initialization
                        print("Successfully intialize the following modules")
                        print("="*50)
                        module_count = 0
                        for k in org_sdict:
                            if ((org_sdict[k]-new_sdict[k])).sum()!=0:
                                print(k)
                                module_count += 1
                        print("="*50)
                        print(f"{module_count}/{len(org_sdict)} module are loaded")
                        model = model.cuda()
                        model.eval()

                        db_names = {
                            1:["0.01-0.1","0.1-1.0","1.0-10.0","10.0-100.0"],
                            2:["0.01-0.1","0.1-1.0","1.0-10.0","10.0-100.0"],
                            3:["0.01-0.1","0.1-1.0","1.0-10.0","10.0-100.0"],
                            6:["type1","type2","type3"]
                        }[task_num]
                        
                        db_score = dict()
                        for db_name in db_names:
                            dataset, target_vocab = get_dataset(
                                tokenizer=tokenizer,
                                file_path = f"../data/curated/{task_num}/{method.split('-')[0]}-{'all' if 'clz' not in method else model_name}/test/{uom}-{notation}-{db_name if task_num==6 else 'diff_'+db_name}.bin",
                                method=method,
                                pe_type=pe
                            )
                                    
                            if 'clz' in method:
                                living_vocab_idx = list(itertools.chain(*[tokenizer(tok,add_special_tokens=False)['input_ids'] for tok in target_vocab]))
                                vocab_ignore_idx = [i for i in range(len(tokenizer)) if i not in living_vocab_idx]
                                data_collator = PredDataCollator(tokenizer=tokenizer, method=method, vocab_ignore_idx=vocab_ignore_idx)
                            else:
                                data_collator = PredDataCollator(tokenizer=tokenizer, method=method)

                            data_loader = DataLoader(
                                        dataset,
                                        batch_size=128,
                                        collate_fn=data_collator,
                                        drop_last=False,
                                        )
                            with torch.no_grad():
                                scores = list()
                                for batch in data_loader:
                                    output =  model(**batch,method=method.split("-")[0])
                                    scores.append(output.scores.detach().cpu())
                                db_score[db_name]=torch.cat(scores).float().mean()

                        wandb.init(config={"exp_type":"curated"}, entity="edlab_sjpark", project="NAACL_curated", name=f"{task_name}/{method}/{model_name}-pe{pe}-{notation}/{uom}", reinit=True)
                        wandb.log(db_score)