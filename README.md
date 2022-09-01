# spaCI
Deciphering spatial cellular communications through adaptive graph model

## Installation
Download spaCI
```
git clone https://github.com/QSong-github/spaCI.git
```

## Dataset Setting
```
|spaCI
├── dataset
│     ├── exp_data_LR.csv
│     ├── train_pos_lr_pairs.csv
│     ├── train_neg_lr_pairs.csv
│     ├── teest_lr_pairs.csv
```


## Tutorial for spaCI

## setting parameters in yaml
```
path to spaCI/configure.yml
```

## processing scripts
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

## training and testing
```
python triplet_main_yaml.py
```
The script was training a model and saved the model in /path/to/spaCI/checkpoint/triplet/best_f1.pth

And we saved the prediction in:
/path/to/spaCI/results.csv 

You can change the model saving and results save path in the configure.yml


