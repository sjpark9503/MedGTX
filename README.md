# Multimodal Learning of Knowledge Graph & Text

## Features

### Models
+ Translation based KG encoder + BERT-style LM + LXMERT

### Training
+ Masked LM 
+ Masked Node prediction
+ Training Model from randomly initialized embeddings 
+ Configure parameters with config.json

### Evaluation

### Utilities
+ Save & Load pretrained models

## Waiting for implementation
+ Load pre-trained KG embedding for translation based KG encoder
+ Full WandB integration 
+ Literal Regression Loss 
+ Literal Bucket Prediction Loss\
+ GCN based KG encoder\
+ Downstream tasks _TBD

## Release Notes
v0.1 : prototype of KG-LXMERT
v0.2 : support T/V/T data split, masking only for literals (not for entities) 
