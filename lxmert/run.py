import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
PATH = '/home/edlab-sjpark/experiments/kg_txt_multimodal/lxmert/'
SRC_PATH = PATH+'src/run_pretraining.py'
TRAINING_CONFIG = {
    "model_type":"lxmert",
    "config_name":PATH+"config/config.json",
    "tokenizer_name":"bert-base-uncased",
    "train_data_file":PATH+"data/pretraining/train",
    "output_dir":PATH+"pretrained_models/test",
    "mlm_probability":0.15,
    "block_size":512,
    "do_train":True,
#    "do_eval": False,
    "per_device_train_batch_size":4,
    "learning_rate":1e-3,
    "num_train_epochs":1,
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)