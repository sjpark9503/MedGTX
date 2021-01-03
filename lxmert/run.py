import subprocess
import os

# FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
TASK_NAME = 'binary_retrieval'
DB = 'dxprx'
MODEL_TYPE = 'kg'
CONFIGURE = 'H128_L2,2,4'
UNIFIED = True

# Variables
Var_TASK = {}
Var_MODEL = {'both':'KGenc_LMinit', 'lm':'LMinit', 'kg':'KGenc', 'rand':'Randinit'}
Var_UNIFIED = 'Unified' if UNIFIED else ''
assert MODEL_TYPE in Var_MODEL, "Model not supported"
assert DB in ['px','dxprx'], "DB not supported"
assert TASK_NAME in ['pretrain','binary_retrieval'], "Task not supported"

# Model Name
## <LMinit & KGenc> : both, <LMinit only> : lm, <KGenc only> : kg, <RandomInit> : rand
## Unified(Placeholder) for Abstract Node : True or False
MODEL_NAME = f'{DB}_{Var_UNIFIED}{"Uni" if MODEL_TYPE in ["both","kg"] else "No"}KGenc'
RUN_NAME = f'{Var_MODEL[MODEL_TYPE]}_{CONFIGURE}_{Var_UNIFIED}{DB}'

# Paths
EXP_PATH = os.getcwd()

# [ dxprx : 960 batches / 32 bsize ]
# [ px : / 32 bsize
TRAINING_CONFIG = {
    "model_type":"lxmert",
    "do_train": True,
    "evaluate_during_training": True,
    "do_eval": True,
    "prediction_loss_only":False,
    "overwrite_output_dir":True,
    "mlm_probability": 0.15,
    "block_size": 512,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 4,
    "learning_rate": 5e-4,
    "num_train_epochs": 40,
    "logging_steps": int(1920/20),
    "save_steps": int(1e10),
    "eval_steps": int(1920),
    "task" : TASK_NAME,
    "train_data_file":os.path.join(EXP_PATH,"data/{}/train".format(MODEL_NAME)),
    "eval_data_file": os.path.join(EXP_PATH,"data/{}/valid".format(MODEL_NAME)),
    #"test_data_file": os.path.join(EXP_PATH, "data/{}/test".format(MODEL_NAME)),
    "run_name":f"{TASK_NAME}_{RUN_NAME}"
}

if TASK_NAME == 'pretrain':
    SRC_PATH = os.path.join(EXP_PATH, 'src/run_pretraining.py')
    TRAINING_CONFIG['config_name'] = os.path.join(EXP_PATH, f"config/config_{Var_UNIFIED}{DB}.json")
    TRAINING_CONFIG['tokenizer_name'] = "bert-base-uncased"
    TRAINING_CONFIG['output_dir'] = os.path.join(EXP_PATH,"pretrained_models/{}".format(RUN_NAME))
else:
    SRC_PATH = os.path.join(EXP_PATH, 'src/run_downstream.py')
    TRAINING_CONFIG['model_name_or_path'] = os.path.join(EXP_PATH, f'pretrained_models/{RUN_NAME}')
    TRAINING_CONFIG['output_dir'] = os.path.join(EXP_PATH,f"pretrained_models/{TASK_NAME}_{RUN_NAME}")

TRAINING_CONFIG_LIST = list()
for (k,v) in list(TRAINING_CONFIG.items()):
    if (isinstance(v, bool)):
        if v:
            TRAINING_CONFIG_LIST.append("--{}".format(k))
    else:
        TRAINING_CONFIG_LIST.append("--{}={}".format(k,v))
# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)