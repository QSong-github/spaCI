### Configuration Tutorial

The configure.yml look like the following:
You can modify the hyper parameters to meet you own dataset, and build your own model
```
DATASET:
  NAME: TripletData
  TRAIN_ROOT: ../dataset/triplet.csv
  TEST_ROOT: ../dataset/test_pairs.csv
  PRED_ROOT: ../dataset/test_lr_pairs.csv
  MATRIX_ROOT: ../dataset/exp_data_LR.csv
  ADJ_ROOT: ../dataset/spatial_graph.csv
MODEL:
  NAME: TripletGraphModel
  INPUT_DIM: 4000
  GRAPH_DIM: 4000
  MLP_HID_DIMS: [200,50,20]
  GRAPH_HID_DIMS: [200,50,20]
  SAVE_PATH: checkpoint/triplet/
TRAIN:
  LR: 0.000100
  EPOCHS: 10
  SAVE_PATH: checkpoint/triplet/
  BATCH_SIZE: 2048
TEST:
  SAVE_PATH: checkpoint/triplet/best_f1.pth
  BATCH_SIZE: 2048
  PRED: results/predict.csv
  EMB1: results/embed_ligand.csv
  EMB2: results/embed_receptor.csv
  THRESHOLD: 0.500000
SEED: 10
use_cuda: cuda:0
```
Sometimes, we want to try different settings, and don't want to modify the configure.yml manually. Therefore, we provide a script for you, which can generate a configure using command lines. This is very useful when we tried to find the best hyper parameter combinations. The tutorial is like the following: 
We provided a python script named "configuration.py", which you can use to customize configuration for your own dataset. Details of the arguments are shown below:

* --trainroot: a csv file with the (a, p, n) triplets. This is the defined triplet for training.
* --testroot: a csv file with the ligand-receptor pairs, we have labels for validation. This is the defined LR pairs for validation. You need to provide labels in this csv file.
* --predroot: a csv file to predict the whether a ligand-receptor pair is positive or not. We only have the data, without labels. This is the defined LR pairs for prediction. You don't need to provide labels in this csv file.
* --input_dim: the input dimension for mlp trunk, which should be the dimension of gene expressions, unless you want to do some dimension reduction in your own settings.
* --graph_dim: the input dimension for graph trunk, which should be the dimension of gene expression, unless you want to do some dimension reduction in your own settings.
* --mlp_hid_dims: the hidden dimensions of mlp layers, you can customize your own MLP layers, with arbitrary dimensions for arbitrary number of layers.
* --graph_hid_dims: the hidden dimensions of graph layers,  you can customize your own graph layers, with arbitrary dimensions for arbitrary number of layers.
* --lr: the learning rate, this is the learning rate for SGD/Adam optimizer for training.
* --epochs: the total rounds of training
* --save_path: path for check points
* --batch_size: the number of pairs for each batch
* --test_save_path: the selected checkpoint path
* --pred: the path for saving the prediction in the form of csv files.
* --emb1: the path for saving embedings of ligand in the form of csv files.
* --emb2: the path for saving embedings of receptor in the form of csv files.
* --threshold: the threshold to determine whether a L-R pair is positive or not.
* --seed: the fixed seed.
* --ymlname: name of the configuration yaml file.
* --use_cuda: defined whether you want to use gpu or not. You need to give the device id like "cuda:0" or "cpu"

a demo usage will be:
```
python3 configuration.py --ymlname configure.yml
```
