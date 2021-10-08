import os
import json

class Configuration():
    def __init__(self, config):
        self.config = config
        self.EXP_PATH = os.getcwd()
        self.DB = config['db']
        self.DB_size = 2000 if self.DB=='dx,prx' else 1000
        self.MODEL_TYPE = config['architecture']
        self.Dim_Hidden = 128
        self.NUM_Layers = {'lang':2, 'kg':2, 'cross':4}
        self.Encoder_Type = {'lang':'bilstm' if (config['model']=='lstm') and (config['task_number']!=2) else config['model']}
        # Variables
        self.Var_TASK = {}
        self.Var_MODEL = {'both':'KGenc_LMinit', 'lm':'LMinit', 'kg':'KGenc', 'rand':'Randinit'}
        self.Var_Unified = 'Unified' if config['P'] else ''
        self.Var_Align = 'Align_' if config['A'] else ''
        self.Var_RC = 'RC_' if config['R'] else ''

        # Helper Variables (TASK_POOL, isSingleModel)
        self.TASK_POOL = {0: 'Pre', 
                        1: 'Re',
                        2: 'Gen',
                        3: 'AdmPred',
                        4: 'ErrDetect',
                        5: 'ReAdm',
                        6: 'NextDx',
                        7: 'Death30',
                        8: 'Death180',
                        9: 'Death365'}
        self.TASK_NAME = self.TASK_POOL[config['task_number']]

        self.MODEL_NAME = f'{self.DB}_{self.Var_Unified}{"Uni" if self.MODEL_TYPE in ["both","kg"] else "No"}KGenc'
        self.RUN_NAME = f'{self.DB}/{self.Var_MODEL[self.MODEL_TYPE]}_H{self.Dim_Hidden}_L{self.NUM_Layers["lang"]},{self.NUM_Layers["kg"]},{self.NUM_Layers["cross"]}_{self.Var_Align}{self.Var_RC}{self.Var_Unified}{self.DB_size}'

        self.TRAINING_CONFIG = {
            "seed":config['seed'],
            "model_type":"lxmert",
            "do_train": True,
            "evaluate_during_training": True,
            "do_eval": True,
            "edge_cls": config['R'],
            "align": config['A'],
            "n_negatives": config['n_negatives'],
            "top_k": config['top_k'],
            "prediction_loss_only":False,
            "overwrite_output_dir":False,
            "mlm_probability": 0.15,
            "block_size": 512,
            "train_batch_size": config['train_bsize'],
            "eval_batch_size": config['eval_bsize'],
            "learning_rate": config['lr'],
            "num_train_epochs": config['num_epochs'],
            "num_log_per_epoch": 20,
            "save_per_run": (config['num_epochs']//10) if config['task_number']==0 else int(1e2),
            "num_eval_per_epoch": 2,
            "task" : self.TASK_NAME,
            "knowmix" : config['KnowMix'],
            "use_tpu" : config['use_tpu'],
            "dataloader_pin_memory" : False if not config['use_tpu'] else True,
            "label_domain" : config['label_domain'],
            "train_data_file":os.path.join(f"data/knowmix/{self.DB}_{self.DB_size}/{self.DB}_UnifiedUniKGenc/train"),
            "eval_data_file": os.path.join(f"data/knowmix/{self.DB}_{self.DB_size}/{self.DB}_UnifiedUniKGenc/valid"),
            "test_data_file": os.path.join(f"data/knowmix/{self.DB}_{self.DB_size}/{self.DB}_UnifiedUniKGenc/test"),
            "run_name":f"{self.TASK_NAME}_{self.RUN_NAME}"
        }

    def get_configuration(self):
        SRC_PATH = os.path.join(self.EXP_PATH, 'src/main.py')
        if (self.config['task_number']==0 or self.config['scratch']) and not self.config['evaluation']:
            if self.config['scratch']:
                # SRC_PATH = os.path.join(self.EXP_PATH, 'src/finetune.py')
                self.TRAINING_CONFIG['run_name'] = f"scratch/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}"
                self.TRAINING_CONFIG['output_dir'] = os.path.join(self.EXP_PATH,f"pretrained_models/scratch/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}_RNG{self.config['seed']}")
            else:
                # SRC_PATH = os.path.join(self.EXP_PATH, 'src/pretrain.py')
                self.TRAINING_CONFIG['run_name'] = f"{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}"
                self.TRAINING_CONFIG['output_dir'] = os.path.join(self.EXP_PATH,f"pretrained_models/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}")
            self.TRAINING_CONFIG['tokenizer_name'] = "prajjwal1/bert-{}".format('tiny' if self.Dim_Hidden==128 else 'mini')
            self.TRAINING_CONFIG['config_name'] = os.path.join(self.EXP_PATH, f"config/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}config_H{self.Dim_Hidden}_L{self.NUM_Layers['lang']},{self.NUM_Layers['kg']},{self.NUM_Layers['cross']}_{self.config['model']}_{self.MODEL_TYPE}_{self.Var_Unified}{self.DB}.json")
            with open(os.path.join(self.EXP_PATH, f"config/config_{self.Var_Unified}{self.DB}.json")) as f:
                Config = json.load(f)
            Config['gcn'] = (self.MODEL_TYPE in ['both', 'kg']) and (self.config['model']!='transe')
            if self.config['model']=='transe':
                Config['pretrained_kg_embedding'] = os.path.join(f"data/{self.DB}/{self.Var_Unified}node.bin")
            Config['pretrained_lang_model']['use_weight'] = self.MODEL_TYPE in ['both', 'lm']
            Config['hidden_size'] = self.Dim_Hidden
            Config['intermediate_size'] = 4*self.Dim_Hidden
            Config['pretrained_lang_model']['model_name'] = "prajjwal1/bert-{}".format('tiny' if self.Dim_Hidden==128 else 'mini')
            Config['l_layers'], Config['r_layers'], Config['x_layers'] = (self.NUM_Layers['kg'], self.NUM_Layers['lang'], self.NUM_Layers['cross'])
            Config['token_type_vocab'] = {header:idx for idx, header in enumerate(self.DB.split(','))}
            Config['type_vocab_size']['lang'] = len(Config['token_type_vocab'])
            Config['attention_probs_dropout_prob'] = self.config['dropout']
            Config['hidden_dropout_prob'] = self.config['dropout']
            Config['cross_att_type'] = 'single' if self.config['model']=='single' else 'cross'
            Config['encoder_type'] = self.Encoder_Type
            Config['lit2word_path'] = self.TRAINING_CONFIG['train_data_file'].replace("train","lit2word")
            
            if self.config['scratch'] and self.config['task_number']==2:
                Config['cross_att_type'] = 'unilm'
                Config['max_position_embeddings']['lang'] = 0 if self.Encoder_Type['lang'].lower() in ['bilstm', 'lstm'] else 512
            if self.config['task_number']==2:
                Config['do_eval'] = False
                
            if self.config['task_number']==3:
                for k in self.TRAINING_CONFIG:
                    if 'file' in k:
                        self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/adm')
                Config['num_labels'] = 45 if self.DB=='px' else 95

            elif self.config['task_number']==4:
                for k in self.TRAINING_CONFIG:
                    if 'file' in k:
                        self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/ed')
                self.TRAINING_CONFIG['id2desc'] = os.path.join(self.EXP_PATH,f"data/ed/knowmix/{self.DB}_{self.DB_size}/{self.MODEL_NAME}/id2desc")

            elif self.config['task_number']==5:
                for k in self.TRAINING_CONFIG:
                    if 'file' in k:
                        self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/readm')
                Config['num_labels'] = 1

            elif self.config['task_number']==6:
                for k in self.TRAINING_CONFIG:
                    if 'file' in k:
                        self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/nextdx')
                Config['num_labels'] = 845

            elif self.config['task_number'] in [7,8,9]:
                for k in self.TRAINING_CONFIG:
                    if 'file' in k:
                        self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data',f'data/{self.TASK_NAME}')
                Config['num_labels'] = 1

            if not os.path.isdir(os.path.dirname(self.TRAINING_CONFIG['config_name'])):
                os.makedirs(os.path.dirname(self.TRAINING_CONFIG['config_name']))
            with open(self.TRAINING_CONFIG['config_name'],'w') as g:
                json.dump(Config,g)
                
        else:
            _run_name = f"evaluation/{'scratch' if self.config['scratch'] else 'pretrained'}/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}"
            self.TRAINING_CONFIG['run_name'] = _run_name
            # self.TRAINING_CONFIG['run_name'] = 'evaluation/' + self.TRAINING_CONFIG['run_name'] 
            self.TRAINING_CONFIG['model_name_or_path'] = os.path.join(self.EXP_PATH,f"pretrained_models/{'scratch' if self.config['scratch'] else 'pretrained'}/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}_RNG{self.config['seed']}")
            
            # Setting for Eval
            if self.config['evaluation']:
                # if self.config['task_number']==1:
                    # SRC_PATH = os.path.join(self.EXP_PATH, f'src/evaluation.py')
                if self.config['task_number']==2:
                    self.TRAINING_CONFIG['do_train'] = False
                    self.TRAINING_CONFIG['do_predict'] = True  # To do nothing on masking of data_collator when evaluation
                elif self.config['task_number']==3:
                    for k in self.TRAINING_CONFIG:
                        if 'file' in k:
                            self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/adm')
                    # SRC_PATH = os.path.join(self.EXP_PATH, f'src/adm_lvl_prediction.py')
                self.TRAINING_CONFIG['output_dir'] = os.path.join(self.EXP_PATH,f"eval_output/{'scratch' if self.config['scratch'] else 'pretrained'}/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}_RNG{self.config['seed']}")


            else:
                # SRC_PATH = os.path.join(self.EXP_PATH, 'src/finetune.py')
                self.TRAINING_CONFIG['model_name_or_path'] = os.path.join(self.EXP_PATH,f"pretrained_models/Pre/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}")
                self.TRAINING_CONFIG['run_name'] = f"pretrained/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}"
                self.TRAINING_CONFIG['output_dir'] = os.path.join(self.EXP_PATH,f"pretrained_models/pretrained/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}_RNG{self.config['seed']}")
                # load config
                with open(f"{self.TRAINING_CONFIG['model_name_or_path']}/config.json") as f:
                    Config = json.load(f)
                # add features
                Config['encoder_type'] = self.Encoder_Type
                Config['attention_probs_dropout_prob'] = self.config['dropout']
                Config['hidden_dropout_prob'] = self.config['dropout']
                Config['cross_att_type'] = 'single' if self.config['model']=='single' else 'cross'
                if self.config['model']=='transe':
                    Config['pretrained_kg_embedding'] = ""
                if self.config['task_number']==2:
                    Config['cross_att_type'] = 'unilm'
                    Config['max_position_embeddings']['lang'] = 0 if self.Encoder_Type['lang'].lower() in ['bilstm', 'lstm'] else 512
                elif self.config['task_number']==3:
                    for k in self.TRAINING_CONFIG:
                        if 'file' in k:
                            self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/adm')
                    Config['num_labels'] = 45 if self.DB=='px' else 95
                elif self.config['task_number']==4:
                    for k in self.TRAINING_CONFIG:
                        if 'file' in k:
                            self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/ed')
                    self.TRAINING_CONFIG['id2desc'] = os.path.join(self.EXP_PATH,f"data/ed/knowmix/{self.DB}_{self.DB_size}/{self.MODEL_NAME}/id2desc")
                elif self.config['task_number']==5:
                    for k in self.TRAINING_CONFIG:
                        if 'file' in k:
                            self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/readm')
                    Config['num_labels'] = 1

                elif self.config['task_number']==6:
                    for k in self.TRAINING_CONFIG:
                        if 'file' in k:
                            self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data','data/nextdx')
                    Config['num_labels'] = 845

                elif self.config['task_number'] in [7,8,9]:
                    for k in self.TRAINING_CONFIG:
                        if 'file' in k:
                            self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace('data',f'data/{self.TASK_NAME}')
                    Config['num_labels'] = 1

                # overwrite config
                if not os.path.isdir(f"config/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.DB}"):
                    os.makedirs(f"config/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.DB}")
                with open(f"config/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}.json",'w') as g:
                    json.dump(Config,g)
                self.TRAINING_CONFIG['config_name'] = os.path.join(self.EXP_PATH,f"config/{self.TASK_NAME}/{'KnowMix,{}/'.format(self.config['KnowMix']) if self.config['KnowMix'] else ''}{self.config['model']}/{self.RUN_NAME}.json")
            
        TRAINING_CONFIG_LIST = list()
        for (k,v) in list(self.TRAINING_CONFIG.items()):
            if (isinstance(v, bool)):
                if v:
                    TRAINING_CONFIG_LIST.append("--{}".format(k))
            else:
                TRAINING_CONFIG_LIST.append("--{}={}".format(k,v))
        return SRC_PATH, TRAINING_CONFIG_LIST

    def assertion(self):
        # if self.config['seed'] not in [1234, 1, 12, 123, 42]:
        #     return False, "Seed out of range"
        if os.path.isdir(self.TRAINING_CONFIG['output_dir']):
            return False, "Output dir"
        elif "adm" in self.config['KnowMix'] and "literal" in self.config['KnowMix']:
            return False, "Choose one of knowledge mixup strategy --> adm or literal"
        elif self.config['architecture'] not in self.Var_MODEL:
            return False, "Model not supported"
        elif self.config['model'] not in ['cross','single','lstm','transe']:
            return False, "Model not supported"
        # elif self.config['KnowMix'] and self.config['architecture'] not in ['both', 'kg']:
        #     return False, "Knowledge mixup layers are only valid with GAT."
        elif self.config['db'] not in ['px','dx,prx']:
            return False, "DB not supported"
        elif self.config['task_number'] not in self.TASK_POOL:
            return False, "Task not supported"
        elif (self.config['A'] is True or self.config['R'] is True) and self.config['scratch']:
            return False, "Scratch start downstream task must turn off alignment prediction & relation classification"
        elif (self.config['architecture'] in ['both','kg']) and self.config['model']=='transe':
            return False, "TransE should turn off KGencoder"
        elif (self.config['task_number']==0) and (self.config['scratch']):
            return False, "Pretrain or Scratch, not both"
        elif (self.config['architecture']!='kg') and self.config['model']=='lstm':
            return False, "LSTM should use KGencoder only"
        else:
            return True, None
