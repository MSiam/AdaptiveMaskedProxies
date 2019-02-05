
# Adaptive Masked Imprinted Weights for Few Shot Segmentation

## Environment setup

```
virtualenv --system-site-packages -p python3 ./venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pre-Trained Weights

Download trained weights [here](https://drive.google.com/drive/folders/1wJXetJCGkT_xej8Jr8Mrj9vJUHN8EtJu?usp=sharing)

## Train on Large Scale Data

```
python train.py --config configs/fcn8s_pascal.yaml
```

## Test few shot setting 

```
python fewshot_imprinted.py --binary BINARY_FLAG --config configs/fcn8s_pascal_imprinted.yml --model_path MODEL_PATH --out_dir OUT_DIR
```
* MODEL_PATH: path for model trained on same fold testing upon.
* OUT_DIR: output directory to save visualization if needed. (optional)
* BINARY_FLAG: 0: evaluates on 17 classes (15 classes previously trained+Bg+New class), 1: evaluate binary with OSLSM method, 2: evaluates binary using coFCN method.
## Configuration
* arch: dilated_fcn8s | fcn8s | reduced_fcn8s
* lower_dim: True (uses 256 nchannels in last layer) | False (uses 4096)
* weighted_mask: True (uses weighted avg pooling based on distance transform)| False (uses mased avg pooling)
* use_norm: True (normalize embeddings during inference)| False
* use_norm_weights: True (normalize extracted embeddings) | False
* use_scale: False: True (Learn scalar hyperaparameter) | False
* dataset: pascal5i (few shot OSLSM setting)| pascal
* fold: 0 | 1 | 2 | 3
* k_shot: 1 | 5

## Visualize predictions and support set
```
python vis_preds.py VIS_FOLDER
```

## Guide to Reproducing Experiments in the paper
Check Experiments.md

Based on semantic segmentation repo:
[SemSeg](https://github.com/meetshah1995/pytorch-semseg)

