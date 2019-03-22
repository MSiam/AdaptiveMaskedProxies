#python cl_naive.py --config configs/fcn8s_pascal_cl.yml --model_path runs/fcn8s_pascalipascal_reduced/reduced_fcn8s_ipascal_best_model.pkl --cl_log cl_results/1cl_naive.txt --niters 1
#python cl_naive.py --config configs/fcn8s_pascal_cl.yml --model_path runs/fcn8s_pascalipascal_reduced/reduced_fcn8s_ipascal_best_model.pkl --cl_log cl_results/2cl_naive.txt --niters 10
python cl_imprinting.py --config configs/fcn8s_pascal_cl.yml --model_path runs/ipascal_reduced/reduced_fcn8s_ipascal_best_model.pkl --cl_log cl_results/3cl_imprinting.txt --alpha 0.05
python cl_imprinting.py --config configs/fcn8s_pascal_cl.yml --model_path runs/ipascal_reduced/reduced_fcn8s_ipascal_best_model.pkl --cl_log cl_results/4cl_imprinting.txt --alpha 0.2
python cl_imprinting.py --config configs/fcn8s_pascal_cl.yml --model_path runs/ipascal_reduced/reduced_fcn8s_ipascal_best_model.pkl --cl_log cl_results/5cl_imprinting.txt --alpha 0.5
python cl_imprinting.py --config configs/fcn8s_pascal_cl.yml --model_path runs/ipascal_reduced/reduced_fcn8s_ipascal_best_model.pkl --cl_log cl_results/6cl_imprinting.txt --alpha 0.9
