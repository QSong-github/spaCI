# spaCI manual and tutorial

### tutorial
  To run spaCI, the input data are shown in the required structure and the data path is assigned in the configure.yml files. Please refer to our preprocess manual [here](https://github.com/QSong-github/spaCI/blob/main/tutorials/tutorial_preprocess.md) for details.
  
With the data and configuration yaml file, spaCI will output embeddings and predictions in one command line:


```python
* python main_yaml.py
```

### manual
  * The following manual shows how spaCI runs step by step:


```python
1. Import python modules
2. Load configurations
3. Fix seed
4. Build dataloader
5. Train model
```

### 1. Import python modules


```python
import argparse

# insert the parent dir into the path
import sys
sys.path.append('../')
import yaml
from model.dataloader import TripletData
from model.model import TripletGraphModel
import os
import torch.utils.data as Data
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
```

### 2. Load configurations
The configure.yml defines the model structure and hypter-parameters for training
You can also manunally change the configure.yml or generate it using the following command:


```python
cmd = 'python ../configuration.py --trainroot ../dataset/triplet.csv --testroot ../dataset/test_pairs.csv --predroot 
../dataset/test_lr_pairs.csv --matrixroot ../dataset/exp_data_LR.csv --adjroot ../dataset/spatial_graph.csv 
--ymlname ../configure.yml --threshold 0.5'
os.system(cmd)
```




    0




```python
yaml_file = '../configure.yml'
with open(yaml_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
```

### 3. Fix seed
All seeds are fixed for reproducibility.
When using GPUs, it is necessary to fix cuda as follows:
```
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
When using CPU, you may comment these lines.


```python
seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 4. Build dataloader
We set up three different modes (mode=0, 1, 2) when building the dataset
* mode=1: for training step. spaCI is in training mode.
* mode=0: for evaluatation step. Check F1 score on validation set.
* mode=2: for prediction step. Save L-R embeddings and output L-R interaction predictions.

Note that the output of training(mode=1) returns:
* (a, p, n) which refers to the gene triplet anchor, positive and negative pairs
* (aid, pid, nid) are the corresponding index in the csv files

The output of evaluation(mode=0) returns:
* (x1, x2) which referes to the expression of receptor and ligand
* (y) if the label (positive/negative), which is used for evaluation and calculating the accuracy and F1-score
* (x1id, x2id) are the corresponding index in the csv files

The output of prediction(mode=2) returns:
* (x1, x2) refers to the expression of receptor and ligand
* (x1id, x2id) are the corresponding index in the csv files

The difference between prediction mode (mode=2) and evaluation(mode=0) is if ground truth labels exist.


```python
def build_dataset(cfg, train=1):
    if train == 1:
        root = cfg['DATASET']['TRAIN_ROOT']
    elif train == 0:
        root = cfg['DATASET']['TEST_ROOT']
    elif train == 2:
        root = cfg['DATASET']['PRED_ROOT']

    if cfg['DATASET']['NAME'] == 'TripletData':
        dataset = TripletData(istrain=train,
                              dataroot=root,
                              matrixroot=cfg['DATASET']['MATRIX_ROOT'])
    else:
        raise NotImplementedError
    return dataset
```


```python
train_dataset = build_dataset(cfg, train=1)

train_dataloader = Data.DataLoader(train_dataset,
                                batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                shuffle=True)
```


```python
test_dataset = build_dataset(cfg, train=0)

test_dataloader = Data.DataLoader(test_dataset,
                                batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                shuffle=False)
```

### 4.1 Load spatial graph 
In spaCI, the spatial graph is calculated by cell locations and saved in the format of csv files.
Please refer to the pre-processing manual [here](https://github.com/tonyyang1995/spaCI/blob/main/README.md) to generate spatial graph for your own data.


```python
adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
adj = torch.from_numpy(adj.to_numpy()).float()
best_f1 = 0
```

### 5. Run spaCI
The configuration ymal file is provided for users to set model parameters.
* INPUT_DIM: the number of genes in your dataset. This is the input of MLP trunk. In this demo, the dimension is 4000
* GRAPH_DIM: the number of genes in your dataset. This is the input of Graph trunk. In this demo, the dimension is 4000
* MlP_HID_DIMs: the hidden dimensions of MLP layers
* GRAPH_HID_DIMS: the hidden dimensions of graph layers
* SAVE_PATH: the folder path of saved checkpoint
* TripletGraphModel: spaCI model structure

### 5.1 Parameter tuning
We provide one-command bash script for parameter tuning in spaCI, which shows the best parameters for spatial dataset


```python
bash parameter_tuning.sh
```

### 5.2 Train spaCI
Below shows the codes for training spaCI. Here you can customize your own epochs in the yaml files.
  In this toy demo, epochs=10 or epochs=20 is enough.
  We evaluate the model performance in every epoch, and save model parameters with the best f1 scores.
  Note that we only validate on validation set, which is not used in training.


```python
best_f1 = 0
for epoch in tqdm(range(cfg['TRAIN']['EPOCHS'])):
    # train
    for batch, (a, p, n, aid, pid, nid) in enumerate(train_dataloader):
        inputs = {}
        inputs['A'] = a; inputs['P'] = p; inputs['N'] = n
            
        inputs['adj'] = adj
        model.set_input(inputs, istrain=1)
        model.single_update()
    
    f1 = infer(model, cfg, verbose=False)
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        model.save('best_f1')
model.save('final')
```

    100%|███████████████████████████████████████████| 10/10 [01:09<00:00,  6.95s/it]


### 5.3 Show evaluation results on validation set
After training spaCI, the performance of the final model will be print with metrics as below:
* accuracy
* Precision
* Recall
* Specificity
* Sensitivity
* F1-score


```python
f1 = infer(model, cfg, load_model='best_f1', verbose=True)
```

    
    ------------------------results----------------------
           112	       123
           977	       981
          acc:	    0.9864
    precision:	    0.9655
       recall:	    0.9106
    Specificity:	    0.9959
    Sensitivity:	    0.9106
    F1-measure:	    0.9372
    ------------------------------------------------------
    


### 5.4 Inference
* Infer is used to evaluate the performance of spaCI and identify the best model on validation set. When "load_model" was not assign, or is None, we will use the default parameters in the pre-defined "model" object. Otherwise, we will load the saved checkpoint from disk.
* verbose is used to print the evaluation results. If verbose=True, the performance of current model in validation set will be print.

Note that we divided the train/val set. The inference can be used to evaluate the model performance and help us to find the model with best F1-score. You can consider this as a trick for early stop/find best model strategy. 


```python
@torch.no_grad()
def infer(model, cfg, load_model=None, verbose=False):
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = build_dataset(cfg, train=0)
    dataloader = Data.DataLoader(dataset,
                                 batch_size=cfg['TEST']['BATCH_SIZE'],
                                 shuffle=False)
    if load_model is not None:
        model_path = os.path.join(cfg['MODEL']['SAVE_PATH'],
                                  load_model + '.pth')
        model.load(model_path)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    label_tp = 0
    label_tn = 0

    # check dirs
    dirs = cfg['TEST']['PRED']
    dirs = dirs.split('/')[:-1]
    dirs = '/'.join(dirs)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    savepred = open(cfg['TEST']['PRED'], 'w')
    savepred.write('ligand,receptor,truelabel,pred\n')
    adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
    adj = torch.from_numpy(adj.to_numpy()).float()

    for batch, (x1, x2, y, x1id, x2id) in enumerate(dataloader):
        inputs = {}
        inputs['x1'] = x1
        inputs['x2'] = x2
        inputs['label'] = y
        inputs['adj'] = adj
        threshold = cfg['TEST']['THRESHOLD']

        model.set_input(inputs, istrain=0)
        dis = model.inference()
        # print(pred.shape, y.shape)
        dis = dis.detach().cpu()

        pred = torch.zeros(dis.shape)
        pred[dis > threshold] = 1

        TP += ((pred == 1) & (y == 1)).sum()
        TN += ((pred == 0) & (y == 0)).sum()
        FP += ((pred == 1) & (y == 0)).sum()
        FN += ((pred == 0) & (y == 1)).sum()
        label_tp += (y == 1).sum()
        label_tn += (y == 0).sum()

        for i in range(len(x1id)):
            id1, id2 = x1id[i], x2id[i]
            savepred.write('%s,%s,%d,%d,%.4f\n' %
                           (id1, id2, y[i], int(pred[i]), dis[i]))

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    sensitive = TP / (TP + FN) if (TP + FN) else 0
    specity = TN / (TN + FP) if (TN + FP) else 0
    acc = (TP + TN) / (label_tp + label_tn)
    F1 = (2 * precision * recall) / (precision + recall)

    if verbose:
        message = '\n------------------------results----------------------\n'
        message += '{:>10d}\t{:>10d}\n'.format(TP, label_tp)
        message += '{:>10d}\t{:>10d}\n'.format(TN, label_tn)
        message += '{:>10}\t{:>10.4f}\n'.format('acc:', acc)
        message += '{:>10}\t{:>10.4f}\n'.format('precision:', precision)
        message += '{:>10}\t{:>10.4f}\n'.format('recall:', recall)
        message += '{:>10}\t{:>10.4f}\n'.format('Specificity:', specity)
        message += '{:>10}\t{:>10.4f}\n'.format('Sensitivity:', sensitive)
        message += '{:>10}\t{:>10.4f}\n'.format('F1-measure:', F1)
        message += '------------------------------------------------------\n'
        print(message)
    return F1
```

### 5.5 Save L-R embeddings and predictions
In this stage, L-R interactions are predictd and saved with their embeddings.
  Since the model have been trained, and all the parameters are fixed, you can predict from multiple inputs, and and do some down-stream tasks based on the predictions or embeddings.
  The predictions and embedding will be saved in the following path, and you can customize it from configure.yml.
* L-R predictions will be saved in "results/predict.csv"
* L-R embeddings will be saved in "results/embed_ligand.csv" and "results/embed_receptor.csv"


```python
@torch.no_grad()
def predict(cfg, load_model=None):
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_model(cfg)

    dataset = build_dataset(cfg, train=2)
    dataloader = Data.DataLoader(dataset,
                                 batch_size=cfg['TEST']['BATCH_SIZE'],
                                 shuffle=False)
    if load_model is not None:
        model_path = os.path.join(cfg['MODEL']['SAVE_PATH'],
                                  load_model + '.pth')
        model.load(model_path)

    savepred = open(cfg['TEST']['PRED'], 'w')
    savepred.write('ligand,receptor,truelabel,pred\n')

    adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0,
                      index_col=0)  #, chunksize=1000)
    adj = torch.from_numpy(adj.to_numpy()).float()
    threshold = cfg['TEST']['THRESHOLD']

    embs1 = None
    embs2 = None
    index1 = None
    index2 = None

    for batch, (x1, x2, x1id, x2id) in enumerate(dataloader):
        inputs = {}
        inputs['x1'] = x1
        inputs['x2'] = x2
#         inputs['label'] = y
        inputs['adj'] = adj

        model.set_input(inputs, istrain=0)
        dis, emb1, emb2 = model.inference(return_intermediate=True)
        # print(x1id, emb1.shape)
        dis = dis.detach().cpu()
        emb1 = emb1.detach().cpu().numpy()
        emb2 = emb2.detach().cpu().numpy()

        if embs1 is None:
            embs1 = emb1
            index1 = x1id
        else:
            embs1 = np.concatenate([embs1, emb1], axis=0)
            index1 = np.concatenate([index1, x1id], axis=0)

        if embs2 is None:
            embs2 = emb2
            index2 = x2id
        else:
            embs2 = np.concatenate([embs2, emb2], axis=0)
            index2 = np.concatenate([index2, x2id], axis=0)

        pred = torch.zeros(dis.shape)
        pred[dis > threshold] = 1

        for i in range(len(x1id)):
            id1, id2 = x1id[i], x2id[i]
            savepred.write('%s,%s,%d,%.4f\n' %
                           (id1, id2, int(pred[i]), dis[i]))
        df1 = pd.DataFrame(embs1, index=index1)
        df2 = pd.DataFrame(embs2, index=index2)
        df1.to_csv(cfg['TEST']['EMB1'])
        df2.to_csv(cfg['TEST']['EMB2'])

    print('done')
```

The prediction function is shown below. When the prediction is completed, it will print "done".


```python
predict(cfg, load_model='best_f1')
```

    done

