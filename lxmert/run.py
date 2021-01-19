import subprocess
import json
import os
# ======================= CONFIG ==================== #
## GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
## TASK & DB
Evaluation = False
TASK_NAME = 'binary_retrieval'
DB = 'dx,prx'
DB_size = 2000
## Pretraining Configs
MODEL_TYPE = 'both'
Unified = True
Align = False
Relation_Classification = True
Scratch_Downstream = False
## Important Hyperparameters
Dim_Hidden = 128
NUM_Layers = {'lang':2, 'kg':2, 'cross':4}
Dropout = 0.1
Num_Negatives = 1
Margin = 1.0
# ======================= CONFIG ==================== #
# Variables
Var_TASK = {}
Var_MODEL = {'both':'KGenc_LMinit', 'lm':'LMinit', 'kg':'KGenc', 'rand':'Randinit'}
Var_Unified = 'Unified' if Unified else ''
Var_Align = 'Align_' if Align else ''
Var_RC = 'RC_' if Relation_Classification else ''
assert MODEL_TYPE in Var_MODEL, "Model not supported"
assert DB in ['px','dx,prx'], "DB not supported"
assert TASK_NAME in ['pretrain', 'binary_retrieval', 'generation', 'single_pretrain', 'single_binary_retrieval', 'single_generation'], "Task not supported"
if Scratch_Downstream is True:
    assert Align is False and Relation_Classification is False, "Scratch start downstream task must turn off alignment prediction & relation classification"

# Model Name
## <LMinit & KGenc> : both, <LMinit only> : lm, <KGenc only> : kg, <RandomInit> : rand
## Unified(Placeholder) for Abstract Node : True or False
MODEL_NAME = f'{DB}_{Var_Unified}{"Uni" if MODEL_TYPE in ["both","kg"] else "No"}KGenc'
RUN_NAME = f'{DB}/{Var_MODEL[MODEL_TYPE]}_H{Dim_Hidden}_L{NUM_Layers["lang"]},{NUM_Layers["kg"]},{NUM_Layers["cross"]}_{Var_Align}{Var_RC}{Var_Unified}{DB_size}'

# Paths
EXP_PATH = os.getcwd()

TRAINING_CONFIG = {
    "seed":1234,
    "model_type":"lxmert",
    "do_train": True,
    "evaluate_during_training": True,
    "do_eval": True,
    "edge_cls": Relation_Classification,
    "align": Align,
    "n_negatives": Num_Negatives,
    "prediction_loss_only":False,
    "overwrite_output_dir":False,
    "mlm_probability": 0.15,
    "block_size": 512,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 4,
    "learning_rate": 1e-5,
    "num_train_epochs": 20,
    "num_log_per_epoch": 20,
    "save_per_run": 4,
    "num_eval_per_epoch": 2,
    "task" : TASK_NAME,
    "train_data_file":os.path.join(EXP_PATH,f"data/{DB}_{DB_size}/{MODEL_NAME}/train"),
    "eval_data_file": os.path.join(EXP_PATH,f"data/{DB}_{DB_size}/{MODEL_NAME}/valid"),
    "test_data_file": os.path.join(EXP_PATH,f"data/{DB}_{DB_size}/{MODEL_NAME}/test"),
    "run_name":f"{TASK_NAME}_{RUN_NAME}"
}

if (TASK_NAME in ['pretrain', 'single_pretrain']) or Scratch_Downstream:
    if Scratch_Downstream:
        SRC_PATH = os.path.join(EXP_PATH, 'src/finetune.py')
        TRAINING_CONFIG['output_dir'] = os.path.join(EXP_PATH,f"pretrained_models/scratch_{TASK_NAME}/{RUN_NAME}")
    else:
        SRC_PATH = os.path.join(EXP_PATH, 'src/pretrain.py')
        TRAINING_CONFIG['output_dir'] = os.path.join(EXP_PATH,f"pretrained_models/{TASK_NAME}/{RUN_NAME}")
    TRAINING_CONFIG['tokenizer_name'] = "bert-base-uncased"
    TRAINING_CONFIG['config_name'] = os.path.join(EXP_PATH, f"config/config_H{Dim_Hidden}_L{NUM_Layers['lang']},{NUM_Layers['kg']},{NUM_Layers['cross']}_{MODEL_TYPE}_{Var_Unified}{DB}.json")
    with open(os.path.join(EXP_PATH, f"config/config_{Var_Unified}{DB}.json")) as f:
        Config = json.load(f)
    Config['gcn'] = MODEL_TYPE in ['both', 'kg']
    Config['pretrained_lang_model']['use_weight'] = MODEL_TYPE in ['both', 'lm']
    Config['hidden_size'] = Dim_Hidden
    Config['intermediate_size'] = 4*Dim_Hidden
    Config['pretrained_lang_model']['model_name'] = "prajjwal1/bert-{}".format('tiny' if Dim_Hidden==128 else 'mini')
    Config['l_layers'], Config['r_layers'], Config['x_layers'] = (NUM_Layers['kg'], NUM_Layers['lang'], NUM_Layers['cross'])
    Config['token_type_vocab'] = {header:idx for idx, header in enumerate(DB.split(','))}
    Config['type_vocab_size']['lang'] = len(Config['token_type_vocab'])
    Config['margin'] = Margin
    Config['attention_probs_dropout_prob'] = Dropout
    Config['hidden_dropout_prob'] = Dropout
    Config['cross_att_type'] = 'single' if TASK_NAME.split('_')[0] == 'single' else 'cross'
    with open(TRAINING_CONFIG['config_name'],'w') as g:
        json.dump(Config,g)
        
else:
    if Evaluation:
        SRC_PATH = os.path.join(EXP_PATH, 'src/evaluation.py')
        TRAINING_CONFIG['model_name_or_path'] = os.path.join(EXP_PATH, f'pretrained_models/{TASK_NAME}/{RUN_NAME}')
        if TASK_NAME in ['generation']:
            SRC_PATH = os.path.join(EXP_PATH, 'src/evaluation_generation.py')
            TRAINING_CONFIG['decode_option'] = {"perturb_type" : 'pad_all', # init_all, pad_all, None
                                                "given_lang_tokens": 1, # 1,5,25
                                                "clean_outputs": True}
        TRAINING_CONFIG['output_dir'] = os.path.join(EXP_PATH,f"eval_output/{TASK_NAME}/{RUN_NAME}")
    else:
        SRC_PATH = os.path.join(EXP_PATH, 'src/finetune.py')
        if TASK_NAME.split('_')[0] == 'single':
            TRAINING_CONFIG['model_name_or_path'] = os.path.join(EXP_PATH, f'pretrained_models/single_pretrain/{RUN_NAME}')
        else:
            TRAINING_CONFIG['model_name_or_path'] = os.path.join(EXP_PATH, f'pretrained_models/pretrain/{RUN_NAME}')
        TRAINING_CONFIG['output_dir'] = os.path.join(EXP_PATH,f"pretrained_models/{TASK_NAME}/{RUN_NAME}")
        # load config
        with open(f"{TRAINING_CONFIG['model_name_or_path']}/config.json") as f:
            Config = json.load(f)
        # add features
        Config['margin'] = Margin
        Config['attention_probs_dropout_prob'] = Dropout
        Config['hidden_dropout_prob'] = Dropout
        if TASK_NAME in ['generation']:
            Config['cross_att_type'] = 'unilm'
        else:
            Config['cross_att_type'] = 'single' if TASK_NAME.split('_')[0] == 'single' else 'cross'
        # overwrite config
        with open(f"{TRAINING_CONFIG['model_name_or_path']}/config.json",'w') as g:
            json.dump(Config,g)
    
TRAINING_CONFIG_LIST = list()
for (k,v) in list(TRAINING_CONFIG.items()):
    if (isinstance(v, bool)):
        if v:
            TRAINING_CONFIG_LIST.append("--{}".format(k))
    else:
        TRAINING_CONFIG_LIST.append("--{}={}".format(k,v))

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)

