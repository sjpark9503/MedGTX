# Multimodal Learning of Knowledge Graph & Text

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
+ Load pre-trained KG embedding for translation based KG encoder
+ Full WandB integration 
+ Literal Regression Loss 
+ Literal Bucket Prediction Loss
+ GCN based KG encoder
+ Downstream tasks _TBD

## Release Notes
__v0.1__ : prototype of KG-LXMERT
__v0.2__ : support T/V/T data split, masking only for literals (not for entities) 
