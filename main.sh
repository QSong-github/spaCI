#!bin/bash

for K in 2
do
    for p in $(seq 0.01 0.01 0.02)
    do
        for alpha in $(seq 0.8 0.1 0.9)
        do
            # echo "${K}, ${p}, ${alpha}"
            cd src
            /opt/R/4.0.2/bin/Rscript spaCI_preprocess.R ../example_data/st_expression.csv ../example_data/st_meta.csv ${K} ${p}  '../data_IO/real_data'
            cd ..
            python3 find_best_parameters.py --alpha ${alpha} --k ${K} --p ${p}
        done
    done
done
   
# print log name
python3 print_best_parameters.py

