# spaCI
Deciphering spatial cellular communications through adaptive graph model

## Installation
Download spaCI
```
git clone https://github.com/QSong-github/spaCI.git
```
spaCI is built based on pytorch, tested in: Ubuntu 18.04, 2080TI GPU, Intel i9-9820, 3.30GHZ, 20 core, 64 GB, CUDA environment(cuda 11.2)

## Tutorial and detailed manual
1. Generate configuration files [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_conf.ipynb)
2. Use the script "parameter_tuning.sh" to find the best parameters:
```
bash parameter_tuning.sh
```
3. Train spaCI [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_train.ipynb)

## Dataset Setting
1. Dataset folder
```
|spaCI
├── data_IO
│     ├── exp_data_LR.csv
│     ├── triplet.csv
│     ├── test_pairs.csv
│     ├── test_lr_pairs.csv
│     ├── spatial_graph.csv
```

Optional: generate spatial graph from real data
```
|spaCI
├── example_data
│     ├── st_expression.csv
│     ├── st_meta.csv
```

2. Setting parameters in yaml
```
path to spaCI/configure.yml
```

### Model training and prediction
```
python main_yaml.py
```
The script was training a model and saved the model in /path/to/spaCI/checkpoint/triplet/best_f1.pth

The inferred ligand-receptor interactions are saved by default in:
/path/to/spaCI/results/spaCI_prediction.csv 

The path of saved model and results can be changed in the configure.yml

## optional 
For your own dataset, you need to prepare the following files:      
(1) a gene expression matrix.     
(2) a pair file with three columns: gene1, gene2 and label (1: interaction; or 0: non-interaction).

We have provided the preprocessing scripts for genearting required data structure for spaCI: 

### Processing scripts
```
python preprocessing.py
```
or:
```
cd src
Rscript spaCI_preprocess.R
```
