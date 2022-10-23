from operator import index
import numpy as np 
import csv, torch, os, pickle, math  
from torch import nn 
from torch.nn import functional as F 
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
# from tqdm import tqdm 
# from random import random, sample 
# from model import Model
class TripletData(Data.Dataset):
    def __init__(self, istrain=1, dataroot='', matrixroot=''):
        super().__init__()
        self.istrain = istrain
        if istrain == 1:
            root = dataroot
            csv = pd.read_csv(root, header=0, index_col=0)
            anchor = csv['key'].tolist()
            pos = csv['pos'].tolist()
            neg = csv['neg'].tolist()

            self.pair_list = []
            for a, p, n in zip(anchor, pos, neg):
                self.pair_list.append((a,p,n))
        else:
            root = dataroot
            csv = pd.read_csv(root, header=0, index_col=0)
            x1 = csv['ligand'].tolist()
            x2 = csv['receptor'].tolist()
            if istrain == 0:
            	label = csv['label'].tolist()
            else:
            	label = [0 for i in range(len(x1))] 
            self.pair_list = []
            for a, p, n in zip(x1, x2, label):
                self.pair_list.append((a,p,int(n)))
        
        root3 = matrixroot
        lines = open(root3, 'r').readlines()
        key_names = {}
        for line in lines[1:]:
            conts = line.strip().split(',')
            name = conts[0]
            genes = conts[1:]
            genes = [float(g) for g in genes]
            key_names[name] = torch.FloatTensor(genes)
        
        gene_names = {}
        line = lines[0]
        gene_names = line.strip().split(',')[1:]

        self.key_names = key_names
        self.gene_names = gene_names

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        if self.istrain == 1:
            a, p, n = self.pair_list[idx]
            a_feat = self.key_names[a]
            p_feat = self.key_names[p]
            n_feat = self.key_names[n]
            return a_feat, p_feat, n_feat, a, p, n
        
        elif self.istrain == 0:
            x1, x2, label = self.pair_list[idx]
            x1_feat = self.key_names[x1]
            x2_feat = self.key_names[x2]
            label = int(label)
            return x1_feat, x2_feat, label, x1, x2

        elif self.istrain == 2:
            x1, x2, label = self.pair_list[idx]
            x1_feat = self.key_names[x1]
            x2_feat = self.key_names[x2]
            # label = int(label)
            return x1_feat, x2_feat, x1, x2
        # return ligand_feat, receptor_feat, label, ligand_id, receptor_id


class PairData(Data.Dataset):
    def __init__(self, istrain=1):
        super().__init__()
        # root1 = opt.dataroot
        if istrain:
            root1 = 'dataset/cv1/train_pairs_0.5.csv'            

            # read csv root
            csv = pd.read_csv(root1, header=0, index_col=0)
            # print(pos_csv.columns)
            ligand = csv['ligand'].tolist()
            recptor = csv['receptor'].tolist()
            label = csv['label'].tolist()

            self.pair_list = []
            for lig, rec, lab in zip(ligand, recptor, label):
                lab = int(lab)
                self.pair_list.append((lig, rec, lab))

        else:
            root1 = 'dataset/cv1/test_pairs_0.5.csv'            

            # read csv root
            csv = pd.read_csv(root1, header=0, index_col=0)
            # print(pos_csv.columns)
            ligand = csv['ligand'].tolist()
            recptor = csv['receptor'].tolist()
            label = csv['label'].tolist()

            self.pair_list = []
            for lig, rec, lab in zip(ligand, recptor, label):
                lab = int(lab)
                self.pair_list.append((lig, rec, lab))

        # id2feature
        root3 = 'exp_data_LR.csv'
        lines = open(root3, 'r').readlines()
        key_names = {}
        for line in lines[1:]:
            conts = line.strip().split(',')
            name = conts[0]
            genes = conts[1:]
            genes = [float(g) for g in genes]
            key_names[name] = torch.FloatTensor(genes)
            
        # save gene_names
        gene_names = {}
        line = lines[0]
        gene_names = line.strip().split(',')[1:]


        self.key_names = key_names
        self.gene_names = gene_names


        # test
        # remove NAN ZFAT 
        # for k1, k2, l in self.pair_list:
        #     if k1 in self.key_names:
        #         continue
        #     else:
        #         print(k1, k2, l)

    def __len__(self):
        return len(self.pair_list)

    
    def __getitem__(self, idx):
        ligand_id, receptor_id, label = self.pair_list[idx]
        # print(ligand_id)
        ligand_feat = self.key_names[ligand_id]
        receptor_feat = self.key_names[receptor_id]

        return ligand_feat, receptor_feat, label, ligand_id, receptor_id
