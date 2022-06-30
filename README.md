# Graph-Text Multi-Modal Pre-training for Medical Representation Learning

This is a official PyTorch implementation of the MedGTX in "__Graph-Text Multi-Modal Pre-training for Medical Representation Learning__" (Sungjin Park, Seongsu Bae, Jiho Kim, Tackeun Kim, Edward Choi, _CHIL 2022_) [PAPER](https://proceedings.mlr.press/v174/park22a/park22a.pdf)

## Experimental Environments

We distribute the experimental enviroments via [Docker](https://www.docker.com)

```
docker pull sjpark9503/dev_env:chil2022_medgtx_env
```

## Usage
+ Code for MIMIC-III pre-processing in `preprocessing`
+ Code for running experiments in `gtx/run_DxPx.py` and `gtx/run_Rx.py` 
  + Please log in to [W&B](https://wandb.ai) for logging results.
+ Pre-trained models in `gtx/pretrained_models/Pre/`

## Citations

```
@inproceedings{park2022medgtx, 
  title={Graph-Text Multi-Modal Pre-training for Medical Representation Learning},
  author={Park, Sungjin and Bae, Seongsu and Kim, Jiho and Kim, Tackeun and Choi, Edward},
  booktitle={The Conference on Health, Inference, and Learning},
  year={2022}
}
```