import pandas as pd
import numpy as np
import yaml
import torch
import os
import random

def seed_everything(seed):
    # seed = 10
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False 

if __name__ == '__main__':
    yaml_file = 'configure.yml'
    with open(yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    seed_everything(cfg['SEED'])

    f = pd.read_csv(cfg['PREPROSS']['file_root'], header=0, index_col=0)
    ligand = f['ligand'].tolist()
    receptor = f['receptor'].tolist()
    label = f['trueLabel'].tolist()

    threshold = cfg['PREPROSS']['split_threshold']

    save_dir = cfg['PREPROSS']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train = open(os.path.join(save_dir, 'train_pairs.csv'), 'w')
    test = open(os.path.join(save_dir, 'test_pairs.csv'), 'w')
    train_count = 1
    test_count = 1
    train.write(',ligand,receptor,label\n')
    test.write(',ligand,receptor,label\n')
    for lig, rec, l in zip(ligand, receptor, label):
        l = int(l)
        if l:
            rand = random.random()
            if rand < threshold:
                train.write('%d,%s,%s,%d\n'%(train_count, lig, rec, l))
                train_count += 1
            else:
                test.write('%d,%s,%s,%d\n'%(test_count, lig, rec, l))
                test_count += 1
        else:
            rand = random.random()
            if rand < threshold:
                train.write('%d,%s,%s,%d\n'%(train_count, lig, rec, l))
                train_count += 1
            else:
                test.write('%d,%s,%s,%d\n'%(test_count, lig, rec, l))
                test_count += 1
    train.close()
    test.close()

    train = pd.read_csv(os.path.join(save_dir, 'train_pairs.csv'))

    ligand = train['ligand'].tolist()
    receptor = train['receptor'].tolist()
    label = train['label'].tolist()

    pos_pairs = {}
    name_set = set()
    for lig, rec, lab in zip(ligand, receptor, label):
        lab = int(lab)
        name_set.add(lig)
        if lab:
            if lig in pos_pairs:
                pos_pairs[lig].append(rec)
            else:
                pos_pairs[lig] = [rec]

    neg_pairs = {}
    for k,v in pos_pairs.items():
        for name in name_set:
            if name == k:
                continue
            if name in v:
                continue
        
            if k in neg_pairs:
                neg_pairs[k].append(name)
            else:
                neg_pairs[k] = [name]

    t = open(os.path.join(save_dir, 'triplet.csv'), 'w')
    t.write(',key,pos,neg\n')
    count = 1
    for k, pvs in pos_pairs.items():
        if k in neg_pairs:
            nvs = neg_pairs[k]
        
            for pv in pvs:
                for nv in nvs:
                    t.write('%d,%s,%s,%s\n'%(count, k,pv,nv))
                    count+=1
    t.close()

    t = open(os.path.join(save_dir, 'triplet.csv'), 'r').readlines()
    print(len(t))
