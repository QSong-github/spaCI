# spaCI
Deciphering spatial cellular communications through adaptive graph model

## Installation
Download spaCI
```
git clone https://github.com/QSong-github/spaCI.git
```

## Tutorial
1. generate configuration files [here](https://github.com/tonyyang1995/spaCI/blob/main/tutorial_conf.ipynb)
2. train spaCI [here](https://github.com/tonyyang1995/spaCI/blob/main/tutorial_train.ipynb)
3. use the script from "main.sh" to find the best parameters:
```
bash main.sh
```

## Dataset Setting
1. training data settings
```
|spaCI
├── data_IO
│     ├── exp_data_LR.csv
│     ├── triplet.csv
│     ├── test_pairs.csv
│     ├── test_lr_pairs.csv
│     ├── spatial_graph.csv
```

2. generate spatial graph from real data
```
|spaCI
├── example_data
│     ├── st_expression.csv
│     ├── st_meta.csv
```

### Setting parameters in yaml
```
path to spaCI/configure.yml
```

### Processing scripts
```
python preprocessing.py
```
To test spaCI, you need two files:     
(1) a gene expression matrix.  
(2) a pair file with two columns: ligand and receptor.   

If you want to train your own dataset, you need to prepare the following files:      
(1) a gene expression matrix.     
(2) a pair file with three columns: gene1, gene2 and label (1: interaction; or 0: non-interaction).

And you can split the data into train/test file.
You can set up the split threshold and the save_dir in the configure.yml. 

### Training and testing
```
python triplet_main_yaml.py
```
The script was training a model and saved the model in /path/to/spaCI/checkpoint/triplet/best_f1.pth

The inferred ligand-receptor interactions are saved by default in:
/path/to/spaCI/results.csv 

The path of saved model and results can be changed in the configure.yml


