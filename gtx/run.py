import subprocess
import json
import os
import time
import itertools
from Run_configs import Configuration

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = {
    # task_number : [0] pretrain / [1] retrieval / [2] generation / [3] adm_lvl_prediction / [4] replacement detection
    'task_number' : 3,
    'db' : 'dx,prx',
    # seed : 1234, 123, 12, 1, 42
    'seed' : 1234,
    # model : cross / single / lstm / transe
    'model' : 'lstm',
    # architecture : both / kg / lm / rand
    'architecture' : 'lm',
    # label domain : graph / text
    'label_domain' : 'text',
    'P' : True,
    'A' : True,
    'R' : True,
    'scratch' : True,
    'evaluation' : False,
    ## Training configs
    'train_bsize' : 8,
    'eval_bsize' : 2,
    'lr' : 1e-5,
    'num_epochs' : 20,
    'top_k' : 5,
    'dropout' : 0.1,
    'n_negatives' : 1,
}

# Sanity check
exp_config = Configuration(config)
RUN_FLAG, error_log = exp_config.assertion()
if not RUN_FLAG: 
    raise ValueError(error_log)

# Run script
SRC_PATH, TRAINING_CONFIG_LIST = exp_config.get_configuration()
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
