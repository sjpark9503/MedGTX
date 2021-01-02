import subprocess
import os

# GPU-IDs which you want to run experiment on
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# Paths
EXP_PATH = os.getcwd()
SRC_PATH = os.path.join(EXP_PATH, 'src/run_pretraining.py')
TASK_NAME = 'dxprx_UnifiedUniKGenc'
RUN_NAME = 'LMinit_KGenc_H128_L2,2,4_Unifiedpx'
# step size =  at batch 8
# step size = 1304 at batch 16
TRAINING_CONFIG = {
    "model_type":"lxmert",
    "config_name":os.path.join(EXP_PATH,"config/config_unifiedpx.json"),
    "tokenizer_name":"bert-base-uncased",
    "do_train": True,
    "evaluate_during_training": True,
    "do_eval": True,
    "prediction_loss_only":False,
    "overwrite_output_dir":True,
    "mlm_probability": 0.15,
    "block_size": 512,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "learning_rate": 5e-4,
    "num_train_epochs": 40,
    "logging_steps": int(2628/20),
    "save_steps": int(1e10),
    "eval_steps": int(2628),
    "task" : "pretrain",
    "train_data_file":os.path.join(EXP_PATH,"data/{}/train".format(TASK_NAME)),
    "eval_data_file": os.path.join(EXP_PATH,"data/{}/valid".format(TASK_NAME)),
    #"test_data_file": os.path.join(EXP_PATH, "data/{}/test".format(TASK_NAME)),
    "run_name":RUN_NAME,
    "output_dir":os.path.join(EXP_PATH,"pretrained_models/{}".format(RUN_NAME)),
}

TRAINING_CONFIG_LIST = list()
for (k,v) in list(TRAINING_CONFIG.items()):
    if (isinstance(v, bool)):
        if v:
            TRAINING_CONFIG_LIST.append("--{}".format(k))
    else:
        TRAINING_CONFIG_LIST.append("--{}={}".format(k,v))
# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)