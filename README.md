# spaCI: deciphering spatial cellular communications through adaptive graph model

Cell-cell communications are vital for biological signaling and play important roles in complex diseases. Recent
advances in single-cell spatial transcriptomics (SCST) technologies allow examining the spatial cell
communication landscapes and hold the promise for disentangling the complex ligand-receptor interactions across
cells. However, due to frequent dropout events and noisy signals in SCST data, it is challenging and lack of
effective and tailored methods to accurately infer cellular communications. Herein, to decipher the cell-to-cell
communications from SCST profiles, we propose a novel adaptive graph model with attention mechanisms named
spaCI. spaCI incorporates both spatial locations and gene expression profiles of cells to identify the active ligandreceptor signaling axis across neighboring cells. 

![Image text](https://github.com/QSong-github/spaCI/raw/main/FIgure%201.png)

## Highlights
* spaCI incoporates both spatial locations and gene expressions of cells for revealing the active ligand-receptor signaling axis across neighboring cells.
* spaCI is able to identify the upstream transcriptional factors mediating the identified ligand-receptor
interactions, which allows gaining further insights into the underlying cellular communications, the
discoveries of disease mechanisms, and effective biomarkers.
* spaCI is developed tailored for spatial transcriptomics and provided available as a ready-to-use opensource software, which demonstrates high accuracy and robust performance over existing methods.

## User Manual and Tutorial
* For the step-by-step tutorial, please refer to the jupyter notebook [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_train.ipynb) 
* We provide a toy demo with one-command bash script for parameter tuning in spaCI, please refer to [here](https://github.com/QSong-github/spaCI/blob/main/parameter_tuning.sh)
* Toy data can be downloaded at [here](https://github.com/QSong-github/spaCI/tree/main/dataset)

## FAQ
* __How can I install spaCI?__       
You can download spaCI from our github link:
  ```
  git clone https://github.com/QSong-github/spaCI.git
  ```
  spaCI is built based on pytorch, tested in Ubuntu 18.04, CUDA environment(cuda 11.2)
  the requirement packages includes:
  ```
  torchvision==0.11.1
  torch==1.6.0
  tqdm==4.47.0
  typing==3.7.4.3
  numpy==1.13.3
  pandas==1.5.1
  PyYAML==6.0
  ```
  or you can also use the following scripts:
  ```
  pip install -r requirements.txt
  ```

* __I want to try the toy demo, can I run spaCI in one command line?__    
  You can use the following commands:
  ```
  python configuration.py --trainroot dataset/triplet.csv --testroot dataset/test_pairs.csv --predroot dataset/test_lr_pairs.csv --matrixroot dataset/exp_data_LR.csv --adjroot dataset/spatial_graph.csv --ymlname conf.yml --threshold 0.9' 
  python main_yaml.py
  ```
  or please refer to our training tutorial [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_train.ipynb)

* __How can I apply spaCI in my own dataset? And how to generate the desired format for spaCI?__         
    We prepare a tutorial for data prepraration [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_preprocess.md)

* __Do I need a GPU for running spaCI?__    
    spaCI is able to run on a standard laptop without GPU. For computational efficiency, we provide options in the configuration.yml file, with the setting use_cuda='cpu' or use_cuda='gpu'.

* __Can I generate my own configuraton file using command line?__    
    To enable users generating their specific configure.yml with different hypterparameters, we provide a script to generate the configuration file. Please refer to [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/manual_configure.md) for details.
