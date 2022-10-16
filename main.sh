#!bin/bash

for K in 2..8
do
    for p in $(seq 0.01 0.01 0.1)
    do
        for alpha in $(seq 0.1 0.1 0.9)
        do
            cd src
            Rscript spaCI_preprocess.R ../example_data/st_expression.csv ../example_data/st_meta.csv ${K} ${p}  '../data_IO/real_data'
            cd ..
            python find_best_parameters.py --alpha ${alpha} --k ${K} --p ${p}
        done
    done
done
   
python print_best_parameters.py

