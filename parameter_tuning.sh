#!bin/bash

for K in $(seq 2 1 8)
do
    for p in $(seq 0.01 0.01 0.1)
    do
        for alpha in $(seq 0.1 0.1 0.9)
        do
            cd src
            /opt/R/4.0.2/bin/Rscript spaCI_preprocess.R "../dataset/example_data/st_expression.csv"  "../dataset/example_data/st_meta.csv" ${K} ${p} 'spaCI_database.RDS' '../dataset/real_data'
            cd ..
            python optimize_parameter.py --alpha ${alpha} --k ${K} --p ${p}
        done
    done
done
   
python print_parameter.py

