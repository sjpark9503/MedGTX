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

### :file_folder: Directory (Briefly)

```
.
├── eda
├── knowemb
│   ├── openke
│   ├── optimization
│   ├── wandb
│   └── train.py
├── lxmert
│   ├── data
│   ├── config
│   └── src
│       ├── utils
│       ├── model.py
│       ├── run_pretraining.py
│       └── trainer.py
│   ├── debugging.ipynb
│   └── run.py
├── preprocessing
│   ├── utils
│   ├── extract_text.ipynb
│   ├── table2triple.ipynb
│   ├── triple2subgraph.ipynb
│   └── README.md
└── README.md
```

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
__v3.0.1__ : Fix run.py, easier configuration via run.py

