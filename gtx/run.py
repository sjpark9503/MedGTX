# Base pkgs
import os
import time
import subprocess
from transformers import AutoTokenizer
# import wandb
# wandb_api = wandb.Api()
# Usr Defined Pkgs
from src.configs.runConfig import Configuration
# Logging tool
from src.utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

# Mode, [Exp] Experiment, [Debug] Debugging, [ExpDeep] Experiment w/ Deepspeed
mode = 'Exp'
assert mode in ['Exp', 'ExpDeep', 'Debug']
 
# Select Device
device = "TPU"
device_id = "2"
if device == "GPU":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

# Experiment Settings
config = {
    # Core settings
    'difficulty' : ["hard,word"],
    'task_number' : [3],
    'data_type':['ctx'],
    'UoM' : ['biomedical','general'], # general biomedical
    'probe' : ['clz-frz-full','bin-frz-full'], 
    'pe_type' : [0,1],
    'model' : ['bert-base-uncased','albert-base-v2','bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12','dmis-lab/biobert-v1.1',''],
    'notation' : ['sci-char', 'char', 'sci', 'word'], # ['sci-char', 'char', 'sci', 'word'],

    # Hyperparams
    'lr' : 3e-5, #2e-5 if "clz" in base_config['probe'] else 5e-5,
    'num_epochs' : 10,
    'train_bsize' : '256',
    'eval_bsize' : '128',
    'eval_frequency' : 0.5,

    # Exp envs
    'fp16' : True,
    'device' : device,
    'mode' : mode,

    # RNG seeds (please do not change order)
    'seed' : [42],
}

"""
List of Vars & Values
    task_number : 
        [1] comparison [2] min/max [3] sort [4] unit conversion [5] word2num [6] val_range [7] uom_validity 
    method : 
        [A] bin [B] clz
    notation : 
        [A] word [B] char [C] sci [D] sci-char [E] sci-spec [F] sci-char-spec
    model : 
        [A] bert-base-uncased (BERT)
        [B] bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12 (BlueBERT)
        [C] dmis-lab/biobert-v1.1 (BioBERT)
        [D] albert-base-v2 (ALBERT)
        [C] EleutherAI/gpt-neo-2.7B (GPT-3, neo)
        [D] microsoft/unilm-base-cased (UniLM)
        [E] (scratch)
    UoM : 
        [A] general [B] biomedical
    seed : 
        [42, 1234, 123, 12, 1]
"""
EXP_LIST = [dict()]
for k,v in config.items():
    temp = list()
    for e in EXP_LIST:
        if isinstance(v, list):
            for _v in v:
                e[k] = _v
                temp.append(e.copy())
                _ = e.pop(k)
        else:
            e[k] = v
            temp.append(e.copy())
    EXP_LIST = temp

# Run Exps
for exp_idx, exp_setting in enumerate(EXP_LIST):
    notifier.critical(f"{exp_idx+1}/{len(EXP_LIST)} exp is running...")
    try:
        # Sanity check
        exp_config = Configuration(exp_setting)
        RUN_FLAG, error_log = exp_config.sanity_check()
        if not RUN_FLAG: 
            raise ValueError(error_log)
        # Run script
        SRC_PATH, TRAINING_CONFIG_LIST = exp_config.get_configuration()
        print(subprocess.run(['accelerate','launch',SRC_PATH]+TRAINING_CONFIG_LIST))
        time.sleep(5)
    except KeyboardInterrupt:
        import sys
        sys.exit()
    except:
        print("")
        continue