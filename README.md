# Multimodal Learning of Knowledge Graph & Text

## Requirements
### Set CUDA ver 10.1
~~~
export CUDA_PATH=/usr/local/cuda-10.1
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
~~~

### Python Labraries
~~~
pip install --upgrade torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers wandb
~~~

## Features

### Models
+ Knowledge graph reprentation
  + Translation base : TransE, RotatE
  + GNN base : -
+ Language model
  + BERT base : BERT, BLUE-BERT
+ Cross modal alignment
  + LXMERT

### Training
+ Pretraining methods
  + Masked LM 
  + Masked node prediction
  + Noise contrastive estimation
+ KG Embedding initialization
  + Random
  + TransE, RotatE

### Evaluation

### Utilities
+ Save & Load pretrained models


## Waiting for implementation
+ GCN based KG encoder __(In Progress)__
+ Downstream tasks _TBD_

## Release Notes
__v0.1__ : prototype of KG-LXMERT\
__v0.2__ : support T/V/T data split, add masking only on literals (not for entities)\
__v0.3__ : support load & initialize pretrained KG embedding, add LxmertForKGTokPredAndMaskedLM in model.py, add NodeMasking_DataCollator, NodeClassification_DataCollator, LiteralRegression_DataCollator (now NCE, literal regression & classification is possible)\
__v1.0__ : Add pretraining, TVT split code\
__v1.0.2__ : Fix bugs in v1.0\
__v1.0.3__ : Fix preprocessing bugs in v1.0.2\
__v1.1__ : Add custom trainer.py, fix triple2subgraph.ipynb _(Debugging...)_\
__v1.1.1__ : _(Debugging...)_\
__v1.2__ : Fix trainer.py\
__v1.2.1__ : Fix preprocessing modules, add text preprocessing ipython\
__v1.3__ : Support evaluation\
__v1.3.1__ : Fix preprocessing ipython, add px/rx/dx parts\
__v1.3.2__ : Add warm start KG encoder from translation based KG embedding, add evaluation metrics, pretrainining & evaluation is good-to-go.\
__v1.4__ : Add evaluation code, add prediction mode on data_collator.py\
__v1.5__ : Add unimodal experiment codes\
__v1.5.1__ : Add re-init to PLM for language part, add viz_attention_heads jupyter notebook\
__v1.6__ : Add class-wise accuracy, integrate all evaluation codes into Evaluation.ipynb\
__v1.6.1__ : Add data mapping class (like entity2id -> id2entity, id2literal)\
__v1.7__ : Attention visualization tool is successfully integrated to Evaluation.ipynb. **This is the final version without graph encoder**\
__v2.0__ : Add graph encoder, Refine codes under preprocessing\
__v2.0.1__ : minor bug fix\
__v2.0.2__ : Code refinement (Disambiguate LM init part, Add model spec to WandB configuration)\
__v2.1__ : Support attention mask for homogeneous graph\
__v2.2__ : Reorganize preprocessing code, fully support experiments until 20201219\
__v2.3__ : Supports Downstream task. Debugging & Full-featured downstream task will be supported at v3.0\
__v3.0__ : Now supports Downstream task (Debugging done)\
__v3.0.1__ : Fix run.py, easier configuration via run.py\
__v3.1-v3.3__ : Support 2 downstream tasks (Retrieval, Generation), Major bug fix (preprocessing, data_collator and obejctive in model.py)\
__v4.0__ : 
+ Re-order file structure
```
│   └── src
│       ├── utils
│       ├── model.py
│       ├── trainer.py
│       ├── pretrain.py <- run_pretraining.py
│       ├── finetune.py <- run_downstream.py
│       ├── evaluation.py <- Run evaluation only (no training process, just predict & eval)
│       └── qualitative_evaulation.ipynb <- Evaluation.ipynb, only for qualitative evaluation
```
+ Add __Hits@k__ and __MRR__ to evaluation.py.
+ __Remove structured_cross attetion option__ from config & model.
+ __Early stopping__ based on evaluation loss, wait 2 evaluation step before stop.
+ __Ignore__ signal from negative samples in Masked LM, Masked Literal prediction\
__v4.0.1__ : Add evaluation only option on run.py\
__v4.0.2__ : Fix generation part for being compatible and prepare evaluation on model.py   

__v4.1__ : Evolve generation model and add an evaluation file for generation part\
__v4.1.1__ : Faster MRR, Hits@k evaluation, skip dataset saving, minor bug fix in model.py when ignoring signal from negative sample for Masked LP, Masked LM\
__v4.1.2__ : add Evaluation file for generation (BLEU), add bleu_all on metrics.py, and ongoing... modify on model.py\
__v4.2__ : Add Admission level prediction codes.\
__v4.2.1__ : fix generation evaluation codes\
__v4.3__ : 
+ Add manual seed & seed list. (Base seed is 1234, additional seeds are 42, 1, 12, 123)
+ Contain Seed number & Scratch Training in RUN_NAME (_please check run.py!_)
+ Add text retrieval (given graph, retrieve text), graph retrieval option (given text, retrieve graph)
+ Supports Admission Level Prediction\
__v4.3.1__ : add helper variables(TASK_POOL, isSingleModel) in run.py, update finetune.py \
__v5.0__ : refine run.py.
```
│   └── pretrained_models
│       ├── pretrain <- pretrained models
│       │   ├── "model"
│       │       ├── "db"
│       ├── pretrained <- downstream, start from pretrained models
│       │   ├── "task", e.g.) retrieval
│       │       ├── "model", e.g.) single
│       │           ├── "db", e.g.) dx, prx
│       └── scratch <- downtream, start from scratch
```

