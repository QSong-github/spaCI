import yaml
from loader.dataloader import TripletData
from loader.modelcpu import PairModel, TripletModel, TripletGraphModel
import os
import torch.utils.data as Data
import torch
import random
import numpy as np
import pandas as pd
import time
from tqdm import tqdm 

def seed_everything(seed):
    # seed = 10
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False 

def build_dataset(cfg, train=1):
    root = cfg['DATASET']['TRAIN_ROOT'] if train else cfg['DATASET']['TEST_ROOT']
    if cfg['DATASET']['NAME'] == 'TripletData':
        dataset = TripletData(istrain=train, dataroot=root, matrixroot=cfg['DATASET']['MATRIX_ROOT'])
    else:
        raise NotImplementedError
    return dataset

def build_model(cfg):
    lr = float(cfg['TRAIN']['LR'])
    if cfg['MODEL']['NAME'] == 'TripletGraphModel':
        model = TripletGraphModel(lr=lr, input_dim=cfg['MODEL']['INPUT_DIM'],
                                graph_dim=cfg['MODEL']['GRAPH_DIM'],
                                mlp_channels=cfg['MODEL']['MLP_HID_DIMS'],
                                graph_channels=cfg['MODEL']['GRAPH_HID_DIMS'],
                                save_path = cfg['MODEL']['SAVE_PATH']
                            )
        return model
    else:
        raise NotImplementedError


@torch.no_grad()
def infer(model, cfg, load_model=None):
    seed_everything(cfg['SEED'])
    
    dataset = build_dataset(cfg, train=0)
    dataloader = Data.DataLoader(dataset, 
                    batch_size=cfg['TEST']['BATCH_SIZE'],
                    shuffle=False)
    if load_model is not None:
        model.load(load_model)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    label_tp = 0
    label_tn = 0

    savepred = open(cfg['TEST']['PRED'], 'w')
    savepred.write('ligand,receptor,truelabel,pred\n')
    
    adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
    adj = torch.from_numpy(adj.to_numpy()).float()

    for batch, (x1, x2, y, x1id, x2id) in enumerate(dataloader):
        inputs = {}
        inputs['x1'] = x1; inputs['x2'] = x2; inputs['label'] = y
        # identity
        # inputs['adj'] = torch.diag(torch.ones(x1.shape[1]))
        # threshold = 0.9

        # random 
        # inputs['adj'] = torch.rand((x1.shape[1], x1.shape[1]))
        # inputs['adj'][inputs['adj'] > 0.5] = 1
        # inputs['adj'][inputs['adj'] <= 0.5] = 0
        # diag = torch.diag(torch.ones(x1.shape[1]))
        # inputs['adj'] += diag
        # inputs['adj'][inputs['adj'] > 1] = 1
        # threshold = 0.7

        #adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
        #adj = torch.from_numpy(adj.to_numpy()).float()
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
            savepred.write('%s,%s,%d,%d,%.4f\n'%(id1, id2, y[i], int(pred[i]), dis[i] ))
    
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    sensitive = TP / (TP + FN) if (TP + FN) else 0
    specity = TN / (TN + FP) if (TN + FP) else 0
    acc = (TP + TN) / (label_tp + label_tn)
    
    F1 = (2 * precision * recall) / (precision + recall)
    message = '\n------------------------results----------------------\n'
    message += '{:>10d}\t{:>10d}\n'.format(TP,label_tp)
    message += '{:>10d}\t{:>10d}\n'.format(TN,label_tn)
    message += '{:>10}\t{:>10.4f}\n'.format('acc:', acc)
    message += '{:>10}\t{:>10.4f}\n'.format('precision:', precision)
    message += '{:>10}\t{:>10.4f}\n'.format('recall:', recall)
    message += '{:>10}\t{:>10.4f}\n'.format('Specificity:', specity)
    message += '{:>10}\t{:>10.4f}\n'.format('Sensitivity:', sensitive)
    message += '{:>10}\t{:>10.4f}\n'.format('F1-measure:', F1)
    message += '------------------------------------------------------\n'
    print(message)
    return F1

def train(cfg):
    dataset = build_dataset(cfg, train=1)
    model = build_model(cfg)
    seed_everything(cfg['SEED'])
    dataloader = Data.DataLoader(dataset, 
                    batch_size=cfg['TRAIN']['BATCH_SIZE'],
                    shuffle=True)
    # infer(model, cfg)
    best_f1 = 0
    best_epoch = 0
    
    print('initialize')
    infer(model, cfg)

    adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
    adj = torch.from_numpy(adj.to_numpy()).float()
    #inputs['adj'] = adj



    for epoch in tqdm(range(cfg['TRAIN']['EPOCHS'])):
        #st = time.time()
        #print(epoch)
        for batch, (a, p, n, aid, pid, nid) in enumerate(dataloader):
            inputs = {}
            inputs['A'] = a; inputs['P'] = p; inputs['N'] = n
            # identity
            # inputs['adj'] = torch.diag(torch.ones(a.shape[1]))

            # random
            # inputs['adj'] = torch.rand((a.shape[1], a.shape[1]))
            # inputs['adj'][inputs['adj'] > 0.5] = 1
            # inputs['adj'][inputs['adj'] <= 0.5] = 0
            # diag = torch.diag(torch.ones(a.shape[1]))
            # inputs['adj'] += diag
            # inputs['adj'][inputs['adj'] > 1] = 1

            #adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
            #adj = torch.from_numpy(adj.to_numpy()).float()
            inputs['adj'] = adj

            model.set_input(inputs, istrain=1)
            model.single_update()
        #print(epoch)
        #f1 = infer(model, cfg)
        #if f1 > best_f1:
        #    best_f1 = f1
        #    best_epoch = epoch
        #    model.save('best_f1')
        #print(time.time() - st)
    infer(model, cfg)

if __name__ == '__main__':
    yaml_file = 'configure.yml'
    with open(yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train(cfg)
