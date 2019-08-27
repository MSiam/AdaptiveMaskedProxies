
for fold in {0..3}
do
    echo 'running fold '$fold' with weights runs/reddilatedfcn8s_pascal/reduced_fcn_fold'$fold ' saving at ' out$fold.txt
    python fewshot_imprinted.py --config configs/redfcn8s_pascal_imprinted.yml --model_path runs/reddilatedfcn8s_pascal/reduced_fcn_fold$fold/dilated_fcn8s_pascal_best_model.pkl --binary 1 --fold $fold --iterations_imp 1 >> logs_1shot/out$fold.txt
    python fewshot_imprinted.py --config configs/redfcn8s_pascal_imprinted.yml --model_path runs/reddilatedfcn8s_pascal/reduced_fcn_fold$fold/dilated_fcn8s_pascal_best_model.pkl --binary 2 --fold $fold --iterations_imp 1 >> logs_1shot_fgbg/out$fold.txt
    python fewshot_imprinted.py --config configs/redfcn8s_pascal_imprinted_5shot.yml --model_path runs/reddilatedfcn8s_pascal/reduced_fcn_fold$fold/dilated_fcn8s_pascal_best_model.pkl --binary 1 --fold $fold >> logs_5shot/out$fold.txt
    python fewshot_imprinted.py --config configs/redfcn8s_pascal_imprinted_5shot.yml --model_path runs/reddilatedfcn8s_pascal/reduced_fcn_fold$fold/dilated_fcn8s_pascal_best_model.pkl --binary 2 --fold $fold >> logs_5shot_fgbg/out$fold.txt
    python fewshot_imprinted_finetune.py --config configs/fcn8s_pascal_imprinted_finetune_5shot.yml --model_path runs/reddilatedfcn8s_pascal/reduced_fcn_fold$fold/dilated_fcn8s_pascal_best_model.pkl --binary 1 --fold $fold >> logs_5shot_ft/out$fold.txt
    python fewshot_imprinted_finetune.py --config configs/fcn8s_pascal_imprinted_finetune_5shot.yml --model_path runs/reddilatedfcn8s_pascal/reduced_fcn_fold$fold/dilated_fcn8s_pascal_best_model.pkl --binary 2 --fold $fold >> logs_5shot_fgbg_ft/out$fold.txt
done
