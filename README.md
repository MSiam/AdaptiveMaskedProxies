
# Adaptive Masked Proxies for Few Shot Segmentation

Implementation used in our paper:
* Adaptive Masked Proxies for Few Shot Segmentation

Accepted in Learning from Limited Labelled Data Workshop in Conjunction with ICLR'19.

* [Workshop Paper](https://openreview.net/forum?id=SkeoV4yZUV)
* [Extended Version](https://arxiv.org/pdf/1902.11123v2.pdf)

## Description
Deep learning has thrived by training on large-scale datasets. However, for continual learning in applications such as robotics, it is critical to incrementally update its model in a sample efficient manner. We propose a novel method that constructs the new class weights from few labelled samples in the support set without back-propagation, relying on our adaptive masked proxies approach. It utilizes multi-resolution average pooling on the output embeddings masked with the label to act as a positive proxy for the new class, while fusing it with the previously learned class signatures. Our proposed method is evaluated on PASCAL-5i dataset and outperforms the state of the art in the 5-shot semantic segmentation. Unlike previous methods, our proposed approach does not require a second branch to estimate parameters or prototypes, which enables it to be used with 2-stream motion and appearance based segmentation networks. The proposed adaptive proxies allow the method to be used with a continuous data stream. Our online adaptation scheme is evaluated on the DAVIS and FBMS video object segmentation benchmark. We further propose a novel setup for evaluating continual learning of object segmentation which we name incremental PASCAL (iPASCAL) where our method has shown to outperform the baseline method.

<div align="center">
<img src="https://github.com/MSiam/AdaptiveMaskedProxies/blob/master/figures/adapproxy.png" width="70%" height="70%"><br><br>
</div>

## Qualitative Evaluation on PASCAL-5i
1-way 1-shot segmentation
 <div class="row">
  <div class="column">
    <img src="https://raw.githubusercontent.com/MSiam/AdaptiveMaskedProxies/master/figures/pred_1.png" alt="" style="width:100%">
  </div>
  <div class="column">
    <img src="https://raw.githubusercontent.com/MSiam/AdaptiveMaskedProxies/master/figures/pred_3.png" alt="" style="width:100%">
  </div>
  <div class="column">
    <img src="https://raw.githubusercontent.com/MSiam/AdaptiveMaskedProxies/master/figures/pred_6.png" alt="" style="width:100%">
  </div>
</div> 

<div class="row">
  <div class="column">
    <img src="https://raw.githubusercontent.com/MSiam/AdaptiveMaskedProxies/master/figures/sprt_1.png" alt="" style="width:100%">
  </div>
  <div class="column">
    <img src="https://raw.githubusercontent.com/MSiam/AdaptiveMaskedProxies/master/figures/sprt_6.png" alt="" style="width:100%">
  </div>
  <div class="column">
    <img src="https://raw.githubusercontent.com/MSiam/AdaptiveMaskedProxies/master/figures/sprt_3.png" alt="" style="width:100%">
  </div>
</div> 


##Qualitative Evaluation on LfW
2-way 1-shot segmentation

## Environment setup

Current Code is tested on torch 0.4.0 and torchvision 0.2.0. 

```
virtualenv --system-site-packages -p python3 ./venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pre-Trained Weights

Download trained weights [here](https://drive.google.com/drive/folders/1wJXetJCGkT_xej8Jr8Mrj9vJUHN8EtJu?usp=sharing)

## Python Notebook
[Demo](https://colab.research.google.com/drive/1yuUg6RuBGf2Z0AKMQRl2FwiSfs9UdSaS#scrollTo=H3aduD_aqlmv)

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
Check [Experiments.md](https://github.com/MSiam/AdaptiveMaskedImprinting/blob/master/Experiments.md)

## Related Repos:
* Based on semantic segmentation repo:
[SemSeg](https://github.com/meetshah1995/pytorch-semseg)
* Pascal5i loader based on OSLSM repo loader:
[OSLSM](https://github.com/lzzcd001/OSLSM)

## References

Please cite our paper if you find it useful in your research

```
@article{DBLP:journals/corr/abs-1902-11123,
  author    = {Mennatullah Siam and
               Boris N. Oreshkin},
  title     = {Adaptive Masked Weight Imprinting for Few-Shot Segmentation},
  journal   = {CoRR},
  volume    = {abs/1902.11123},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.11123},
  archivePrefix = {arXiv},
  eprint    = {1902.11123},
  timestamp = {Tue, 21 May 2019 18:03:37 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1902-11123},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

