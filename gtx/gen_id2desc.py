import os
import torch
from tqdm import tqdm, trange
db_name = 'dx,prx'
size = 2000
DB_type = ['UnifiedNoKGenc','UnifiedUniKGenc', 'NoKGenc','UniKGenc',]
SPLIT = ['train','valid','test']

for db_type in DB_type:
    if "Unified" not in db_type:
        NUM_SPECIAL_TOKENS = 3
        id2label = torch.load(f'data/{db_name}_{size}/{db_name}_{db_type}/id2label')
        label2id = {v:k for k,v in id2label.items()}
        id2entity = {int(line.split('\t')[1])+NUM_SPECIAL_TOKENS:line.split('\t')[0] for line in open(os.path.join(f'data/{db_name}','entity2id.txt')).read().splitlines()[1:]}
        label2entity = {k:id2entity[v] for k,v in label2id.items()}
    else:
        id2entity = {v:k.split('\t')[0] for k,v in torch.load(f'data/{db_name}_{size}/{db_name}_{db_type}/unified_node').items()}
    os.makedirs(f'data_ed/knowmix/{db_name}_{size}/{db_name}_{db_type}', exist_ok=True)
    id2desc = {k:v.split('^^')[0].replace('"','') if '^^' in v else "" for k,v in id2entity.items()}
    print(len(id2desc))
    torch.save(id2desc,f'data_ed/knowmix/{db_name}_{size}/{db_name}_{db_type}/id2desc')