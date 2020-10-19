import logging
import os
import pickle
import time
from tqdm import tqdm
import sys

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)
        self.full_block_size = block_size
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                self.examples = torch.load(cached_features_file)
                logger.info(
                    "Loading features from cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            else:
                logger.info("Creating features from dataset file at %s", directory)
                self.examples = list()
                with open(file_path, encoding="utf-8") as f:
                    tokenized_text = list()
                    for line in tqdm(f,total=119371337,file=sys.__stdout__):
                        text = line.rstrip('\n')
                        tokenized_text += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                        if len(tokenized_text)>block_size:
                            self.examples.append(torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenized_text[:block_size]), dtype=torch.long))
                            tokenized_text = tokenized_text[block_size:]

                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                torch.save(self.examples, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.rstrip('\n') for line in f if (len(line) > 0 and not line.isspace())]

        # print(len(lines))
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)