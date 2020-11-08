import subprocess
import os

# GPU-IDs which you want to run experiment on
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Paths
EXP_PATH = os.getcwd()
SRC_PATH = os.path.join(EXP_PATH, 'src/run_pretraining.py')
TASK_NAME = 'masked_literal_prediction'
RUN_NAME = 'run1'
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
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "learning_rate": 1e-3,
    "num_train_epochs": 20,
    "logging_steps": int(5420/20),
    "save_steps": int(5420/2),
    "eval_steps": int(5420/2),
    "train_data_file":os.path.join(EXP_PATH,"data/{}/train".format(TASK_NAME)),
    "eval_data_file": os.path.join(EXP_PATH,"data/{}/valid".format(TASK_NAME)),
    "run_name":RUN_NAME,
    "output_dir":os.path.join(EXP_PATH,"pretrained_models/{}".format(RUN_NAME)),
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)