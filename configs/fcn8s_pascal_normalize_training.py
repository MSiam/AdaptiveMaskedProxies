model:
    arch: reduced_fcn8s
    lower_dim: True
    weighted_mask: False
    use_norm: False
    offsetting: False
    use_normalize_train: True
    use_scale: True
data:
    dataset: pascal
    fold: 0
    n_classes: 16
    train_split: train_aug
    val_split: val
    img_rows: 'same'
    img_cols: 'same'
    path: /usr/work/menna/VOCdevkit/VOC2012/
training:
    train_iters: 300000
    batch_size: 1
    val_interval: 500
    n_workers: 0
    print_interval: 50
    optimizer:
        name: 'rmsprop'
        lr: 1.0e-6
        weight_decay: 0.0005
#        momentum: 0.99
    loss:
        name: 'cross_entropy'
        size_average: False
        pad: False
    lr_schedule:
    resume: fcn8s_pascal_best_model.pkl
