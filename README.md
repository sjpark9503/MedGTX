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
+ Full WandB integration __(In Progress)__
+ Preprocessing Labels for Negative sampling, Bucketized literals & Regression __(In Progress)__
+ GCN based KG encoder
+ Downstream tasks _TBD_

## Release Notes
__v0.1__ : prototype of KG-LXMERT\
__v0.2__ : support T/V/T data split, add masking only on literals (not for entities)\
__v0.3__ : support load & initialize pretrained KG embedding, add LxmertForKGTokPredAndMaskedLM in model.py, add NodeMasking_DataCollator, NodeClassification_DataCollator, LiteralRegression_DataCollator (now NCE, literal regression & classification is possible)\
__v1.0__ : Full-featured (Preprocessing, Pre-training and TVT split evaluation)\
__v1.0.2__ : Debugging done
