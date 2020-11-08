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
  + Translation base : TransE, ComplEx
  + GNN base : -
+ Language model
  + BERT base : BERT, BLUE-BERT, ...
+ Cross modal alignment
  + LXMERT

### Training
+ Pretraining methods
  + Masked LM 
  + Masked node prediction
  + Noise contrastive estimation
+ Embedding initialization
  + Random

### Evaluation

### Utilities
+ Save & Load pretrained models

## Waiting for implementation
+ Full WandB integration __(In Progress)__
+ Preprocessing Labels for Negative sampling, Bucketized literals & Regression __(In Progress)__
+ GCN based KG encoder
+ Downstream tasks _TBD_

## Release Notes
__v0.1__ : prototype of KG-LXMERT\
__v0.2__ : support T/V/T data split, add masking only on literals (not for entities)\
__v0.3__ : support load & initialize pretrained KG embedding, add LxmertForKGTokPredAndMaskedLM in model.py, add NodeMasking_DataCollator, NodeClassification_DataCollator, LiteralRegression_DataCollator (now NCE, literal regression & classification is possible)\
__v1.0__ : Full-featured (Preprocessing, Pre-training and TVT split evaluation)\
__v1.0.2__ : Debugging done\
__v1.0.3__ : Fix triple2subgraph.ipynb\
__v1.1__ : Add custom trainer.py, fix triple2subgraph.ipynb _(Debugging...)_\
__v1.1.1__ : Fix trainer.py _(Debugging...)_\
__v1.2__ : Debugging for trainer.py **done**!\
__v1.2.1__ : Fix preprocessing modules, add text preprocessing ipython\
__v1.3__ : Support evaluation\
__v1.3.1__ : Fix preprocessing ipython, add px/rx/dx parts
