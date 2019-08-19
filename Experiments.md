# Experimental Setup

* Download [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
* Download [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html) + Use [train.txt](http://home.bharathh.info/pubs/codes/SBD/train_noval.txt) provided by SBD instead of PASCAL.
* Modify config.json path to your SBD_PATH
* In all Experiments modify in the config files "path" to the PASCAL_VOC_PATH

## Reproducing experiments in Table 1, 2, 3

1. Unzip runs.zip in the same code folder

2. Run:
```
./run_exps.sh
```
3. Saved logs corresponding to each experiment:
    * logs_1shot: AMP-2 in Table 1
    * logs_5shot: AMP-2 in Table 2
    * logs_1shot_fgbg: AMP-2 in Table 3 1-shot
    * logs_5shot_fgbg: AMP-2 in Table 3 5-shot
    * logs_5shot_fgbg: AMP-2+FT in Table 3 5-shot

## Reproducing iPASCAL results 

1. Checkout the branch for continual learning mode **pascal_multirun**
2. Run:
```
./run_exps_cl.sh
```

3. Plot the results
```
python plot_cl_multiruns.py cl_results_
```

## Running Hyperparameter Search

Based on Tune, follow the instructions for installation [here](https://ray.readthedocs.io/en/latest/tune.html)
```
git checkout hyperopt
python hyperparam_search.py --config CONFIG_FILE --model_path PATH --binary BINARY_FLAG
```

