
# Adaptive Masked Imprinted Weights for Few Shot Segmentation

## Train on Large Scale Data

```
python train.py --config configs/fcn8s_pascal.yaml
```

## Test few shot setting 

```
python few_shot_imprinted.py --binary --config configs/fcn8s_pascal_imprinted.yml --model_path MODEL_PATH --out_dir OUT_DIR
```
* MODEL_PATH: path for model trained on same fold testing upon.
* OUT_DIR: output directory to save visualization if needed. (optional)

## Configuration
* arch: dilated_fcn8s | fcn8s
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

Based on semantic segmentation repo:
[SemSeg](https://github.com/meetshah1995/pytorch-semseg)

