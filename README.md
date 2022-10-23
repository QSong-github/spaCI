# spaCI: deciphering spatial cellular communications through adaptive graph model
Ziyang Tang, Tonglin Zhang, Baijian Yang, Jing Su, Qianqian Song

Cell-cell communications are vital for biological signaling and play important roles in complex diseases. Recent
advances in single-cell spatial transcriptomics (SCST) technologies allow examining the spatial cell
communication landscapes and hold the promise for disentangling the complex ligand-receptor interactions across
cells. However, due to frequent dropout events and noisy signals in SCST data, it is challenging and lack of
effective and tailored methods to accurately infer cellular communications. Herein, to decipher the cell-to-cell
communications from SCST profiles, we propose a novel adaptive graph model with attention mechanisms named
spaCI. spaCI incorporates both spatial locations and gene expression profiles of cells to identify the active ligandreceptor signaling axis across neighboring cells. Through benchmarking with currently available methods, spaCI
shows superior performance on both simulation data and real SCST datasets. Furthermore, spaCI is able to
identify the upstream transcriptional factors mediating the active ligand-receptor interactions. For biological
insights, we have applied spaCI to the seqFISH+ data of mouse cortex and the NanoString CosMx SMI data of
non-small cell lung cancer samples. spaCI reveals the hidden ligand-receptor (L-R) interactions from the sparse
seqFISH+ data, meanwhile identifies the inconspicuous L-R interactions including THBS1−ITGB1 between
fibroblast and tumors in NanoString CosMx SMI data. spaCI further reveals that SMAD3 plays an important role
in regulating the crosstalk between fibroblasts and tumors, which contributes to the prognosis of lung cancer
patients. Collectively, spaCI addresses the challenges in interrogating SCST data for gaining insights into the
underlying cellular communications, thus facilitates the discoveries of disease mechanisms, effective biomarkers,
and therapeutic targets.
![Image text]('https://github.com/QSong-github/spaCI/blob/main/FIgure%201.png', "Figure 1")

## Installation
Download spaCI
```
git clone https://github.com/QSong-github/spaCI.git
```
spaCI is built based on pytorch, tested in Ubuntu 18.04, CUDA environment(cuda 11.2)

## Tutorial and detailed manual
1. Generate configuration file [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_conf.ipynb)
2. Use the script "parameter_tuning.sh" to find the best parameters, i.e., ``` bash parameter_tuning.sh ```
3. Train spaCI model [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_train.ipynb)

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

## Optional 
For your own dataset, you need to prepare the following files:      
(1) Spatial gene expression data    
(2) Spatial cell meta file with cell location information
```
|spaCI
├── example_data
│     ├── st_expression.csv
│     ├── st_meta.csv
```
We have provided the preprocessing scripts for genearting required data structure for spaCI: 

```
python preprocessing.py
```
or:
```
cd src
Rscript spaCI_preprocess.R
```
