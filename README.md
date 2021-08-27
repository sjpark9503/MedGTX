# Multimodal Learning of Knowledge Graph & Text

## Requirements
### Set CUDA ver 11.1
~~~
export CUDA_PATH=/usr/local/cuda-11.1
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
~~~

### Python Labraries
~~~
pip install --upgrade torch==1.8.1+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers wandb
~~~

## Features

### Models
+ Knowledge graph reprentation
  + Translation base : TransE
  + GNN base : GAT
+ Language model
  + BERT base : BERT-tiny
+ Cross modal alignment
  + GTX