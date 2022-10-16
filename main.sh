#!bin/bash

for K in 2
do
    for p in $(seq 0.01 0.01 0.02)
    do
	for alpha in $(seq 0.1 0.1 0.2)
	do
	    cd src
	    Rscript spaCI_preprocess.R ../example_data/st_expression.csv ../example_data/st_meta.csv ${K} ${p}  '../data_IO'
	    cd ..
	    # # sed 's/^\(THRESHOLD: \).*$/\1'"$thr"'/' configure.yml
	    # # cp $yml configure.yml
	    python3 find_best_parameters.py --alpha alpha --k K --p p
	done
    done
done
   
# print log name

