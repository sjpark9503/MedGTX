import subprocess
import os

# GPU-IDs which you want to run experiment on
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Paths
PATH = ''
EXP_PATH = os.path.join(PATH,'kg_txt_multimodal/lxmert')
SRC_PATH = os.path.join(PATH, 'src/run_pretraining.py')
TASK_NAME = ''
RUN_NAME = ''
#
TRAINING_CONFIG = {
    "model_type":"lxmert",
    "config_name":os.path.join(EXP_PATH,"config/config.json"),
    "tokenizer_name":"bert-base-uncased",
    "do_train": True,
    "evaluate_during_training": True,
    "do_eval": True,
    "prediction_loss_only":True,
    "mlm_probability": 0.15,
    "block_size": 512,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "learning_rate": 1e-3,
    "num_train_epochs": 10,
    "logging_steps": 10,
    "save_steps": 10,
    "eval_steps": 10,
    "train_data_file":os.path.join(EXP_PATH,"data/{}/train".format(TASK_NAME)),
    "eval_data_file": os.path.join(EXP_PATH,"data/{}/valid".format(TASK_NAME)),
    "run_name":RUN_NAME,
    "output_dir":os.path.join(EXP_PATH,"pretrained_models/{}".format(RUN_NAME)),
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)