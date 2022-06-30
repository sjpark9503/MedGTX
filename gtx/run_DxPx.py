import subprocess
import json
import os
import time
import itertools
from Run_configs import Configuration
 
# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
os.environ["TOKENIZERS_PARALLELISM"] = 'true' 
 
# TPU setting
TPU = False

for preset in [
    {'db':'dx,prx','model':'cross','architecture':'both','knowmix':'summary','scratch':False, 'unimodal':'', 'note':'reproduce'},
]:
    for _task in [0,1,3,4,5,7,2]:
        if (_task==3) and (preset['db']=='px'):
            continue
        for _SEED in [42]: # , 123, 12, 1, 42]: # , 1, 42]:
            if (_task==0) and (_SEED!=1234):
                continue
            config = {
                # task_number : [0] pretrain / [1] retrieval / [2] generation / [3] adm_lvl_prediction / [4] replacement detection
                #                [5] readmission prediction [6] next admission Dx prediction [7,16,9] Death 30,1160,365
                'task_number' : _task,
                # db: dx,prx / px
                'db' : preset['db'],
                # seed : 1234, 123, 12, 1, 42
                'seed' : _SEED, #1234,
                # model : cross / single / lstm / transe
                'model' : preset['model'],
                # unimodal : graph / text / ""(multimodal)
                'unimodal' : preset['unimodal'],
                # architecture : both / kg / lm / rand
                'architecture' : preset['architecture'],
                # label domain : graph / text
                'label_domain' : 'text',
                'P' : True,
                'A' : not preset['scratch'],
                'R' : False if preset['db']=='px' else True,
                'KnowMix' : preset['knowmix'], # layer, init, adm
                'scratch' : preset['scratch'],
                'evaluation' : False,
                'top_k' : 10,
                'note' : preset['note'],
                'dropout' : 0.1,
                'n_negatives' : 1,
                'use_tpu' : TPU,
            }
            # Training configs
            if _task == 0:
                config['train_bsize'] = 16 if preset['db']=='px' else 32
                config['eval_bsize'] = 4 if preset['db']=='px' else 8
                config['lr'] = 1e-4
                config['num_epochs'] = 40
            elif _task == 2:
                config['train_bsize'] = 16 if preset['db']=='px' else 32
                config['eval_bsize'] = 4 if preset['db']=='px' else 8
                config['lr'] = 3e-5
                config['num_epochs'] = 30
            elif _task in [1,3,4]:
                config['train_bsize'] = 16 if preset['db']=='px' else 32
                config['eval_bsize'] = 4 if preset['db']=='px' else 8
                config['lr'] = 1e-5
                config['num_epochs'] = 30
            elif _task in [5,6,7]:
                config['train_bsize'] = 16 if preset['db']=='px' else 32
                config['eval_bsize'] = 4 if preset['db']=='px' else 8
                config['lr'] = 2e-5
                config['num_epochs'] = 20
            
            # Run script
            exp_config = Configuration(config)
            SRC_PATH, TRAINING_CONFIG_LIST = exp_config.get_configuration()

            # Sanity check
            RUN_FLAG, error_log = exp_config.assertion()
            if not RUN_FLAG: 
                print(error_log)
                continue

            # Bash run
            subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
