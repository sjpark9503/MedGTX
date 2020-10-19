from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple
import os
import torch

def build_dataset(path, train_bs, eval_bs):
    train_path = os.path.join(path, "train")
    validation_path = os.path.join(path, "valid")
    test_path = os.path.join(path, "test")

    train_dataset = torch.load(train_path).long()
    train_dataloader = DataLoader(train_dataset,batch_size=train_bs,shuffle=True)
    if os.path.isfile(validation_path):
        valid_dataset = torch.load(validation_path).long()
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_bs, shuffle=False)
        test_dataset = torch.load(test_path).long()
        test_dataloader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False)
    else:
        valid_dataloader = None
        test_dataloader = None

    voca = {'node':{k:v for (v,k) in enumerate(open(os.path.join(path, "node_dict")).read().splitlines())},
            'edge':{k:v for (v,k) in enumerate(open(os.path.join(path, "edge_dict")).read().splitlines())}}

    return train_dataloader, valid_dataloader, test_dataloader, voca