
# Adaptive Masked Imprinted Weights for Few Shot Segmentation

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

## Code Structure 

* Data Loaders:
  * pascal_voC_loader.py : loader used in training on large scale data on fold x, classes from Ltest are set to 250 (ignored).
  * pascal_voc_5i_loader.py : loader used in fewshot setting, uses exact loader from oslsm in loader/oslsm/ss_datalayer.py.
* Models:
  * dilated_fcn8s: FCN8s with dilated convolution and removed last 2 pooling layers.
  * fcn8s: original FCN8s architecture without padding.
* Imprinting Functionality: Inside each model 3 methods need to exist.
  * imprint: computes the imprinted weights.
  * extract: extract the embeddings using masked/weighted avg pooling.
  * reverse_imrpint: reset weights to original weights for the next samples support set.
* Imprinting Utilities: inside ptsemseg/models/utils.py
  * compute_weights: based on adaptive formulation inspirign from adaptive correlation filters.
  * masked_embeddings: masked avg pooling based on class index

Based on semantic segmentation repo:
[SemSeg](https://github.com/meetshah1995/pytorch-semseg)

