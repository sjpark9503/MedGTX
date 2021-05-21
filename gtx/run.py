import subprocess
import json
import os
import time
import itertools
from Run_configs import Configuration

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# TPU setting
TPU = False

config = {
    # task_number : [0] pretrain / [1] retrieval / [2] generation / [3] adm_lvl_prediction / [4] replacement detection
    'task_number' : 0,
    'db' : 'dx,prx',
    # seed : 1234, 123, 12, 1, 42
    'seed' : 1234,
    # model : cross / single / lstm / transe
    'model' : 'single',
    # architecture : both / kg / lm / rand
    'architecture' : 'both',
    # label domain : graph / text
    'label_domain' : 'text',
    'P' : True,
    'A' : True,
    'R' : True,
    'scratch' : False,
    'evaluation' : False,
    ## Training configs
    'train_bsize' : 16,
    'eval_bsize' : 4,
    'lr' : 1e-4,
    'num_epochs' : 40,
    'top_k' : 5,
    'dropout' : 0.1,
    'n_negatives' : 1,
    'use_tpu' : TPU,
}

# Run script
exp_config = Configuration(config)
SRC_PATH, TRAINING_CONFIG_LIST = exp_config.get_configuration()

# Sanity check
RUN_FLAG, error_log = exp_config.assertion()
if not RUN_FLAG: 
    raise ValueError(error_log)

# Bash run
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
