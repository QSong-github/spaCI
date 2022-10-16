import argparse

import yaml
from model.dataloader import TripletData
from model.model import PairModel, TripletModel, TripletGraphModel
import os
import torch.utils.data as Data
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def build_model(cfg):
    lr = float(cfg['TRAIN']['LR'])
    if cfg['MODEL']['NAME'] == 'TripletGraphModel':
        model = TripletGraphModel(
            lr=lr,
            input_dim=cfg['MODEL']['INPUT_DIM'],
            graph_dim=cfg['MODEL']['GRAPH_DIM'],
            mlp_channels=cfg['MODEL']['MLP_HID_DIMS'],
            graph_channels=cfg['MODEL']['GRAPH_HID_DIMS'],
            save_path=cfg['MODEL']['SAVE_PATH'])
        return model
    else:
        raise NotImplementedError


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

    for batch, (x1, x2, y, x1id, x2id) in enumerate(dataloader):
        inputs = {}
        inputs['x1'] = x1
        inputs['x2'] = x2
        inputs['label'] = y
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
            savepred.write('%s,%s,%d,%d,%.4f\n' %
                           (id1, id2, y[i], int(pred[i]), dis[i]))
        df1 = pd.DataFrame(embs1, index=index1)
        df2 = pd.DataFrame(embs2, index=index2)
        df1.to_csv(cfg['TEST']['EMB1'])
        df2.to_csv(cfg['TEST']['EMB2'])

    print('done')


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


def train(cfg):
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = build_dataset(cfg, train=1)
    model = build_model(cfg)

    dataloader = Data.DataLoader(dataset,
                                 batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                 shuffle=True)
    # infer(model, cfg)
    best_f1 = 0
    best_epoch = 0

    adj = pd.read_csv(cfg['DATASET']['ADJ_ROOT'], header=0, index_col=0)
    adj = torch.from_numpy(adj.to_numpy()).float()

    # print('initialize')
    # infer(model, cfg)
    # print('best_f1:')
    # print(best_f1)

    for epoch in tqdm(range(cfg['TRAIN']['EPOCHS'])):
        for batch, (a, p, n, aid, pid, nid) in enumerate(dataloader):
            inputs = {}
            inputs['A'] = a; inputs['P'] = p; inputs['N'] = n
            
            inputs['adj'] = adj
            model.set_input(inputs, istrain=1)
            model.single_update()
        # print(epoch)
        f1 = infer(model, cfg, verbose=True)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            model.save('best_f1')

        # print ('best_f1:')
        # print (best_f1)
    model.save('final')
    f1 = infer(model, cfg, load_model='best_f1')
    return f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ymlname', type=str, default='configure.yml')
    opt = parser.parse_args()
    yaml_file = opt.ymlname
    # yaml_file = 'configure.yml'
    with open(yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    f1 = train(cfg)
    predict(cfg, load_model='best_f1')

