# spaCI
spatial cell communications through graph attentive model

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
coming soon

## setting parameters in yaml
```
path to spaCI/configure.yml
```

## processing scripts
```
python preprocessing.py
```
To test spaCI, you need two files:
(1) a gene expression matrix
(2) a pair file with two columns: ligend and receptor

If you want to train your own dataset, you need to prepare thhe following files:
(1) a gene expression matrix
(2) a pair file with three columns: ligend , receptor and label (1 or 0)

And you can split the data into train/test file.
You can set up the split threshold and the save_dir in the configure.yml

## training
```
python triplet_main_yaml.py
```
