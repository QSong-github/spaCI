import argparse

def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataname', type=str, default='TripletData')
    parser.add_argument('--trainroot', type=str, default='data_IO/triplet.csv')
    parser.add_argument('--testroot', type=str, default='data_IO/test_pairs.csv')
    parser.add_argument('--predroot', type=str, default='data_IO/test_lr_pairs.csv')
    parser.add_argument('--matrixroot', type=str, default='data_IO/exp_data_LR.csv')
    parser.add_argument('--adjroot', type=str, default='data_IO/spatial_graph.csv')

    # model
    parser.add_argument('--modelname', type=str, default='TripletGraphModel')
    parser.add_argument('--input_dim', type=int, default=4000)
    parser.add_argument('--graph_dim', type=int, default=4000)
    parser.add_argument('--mlp_hid_dims', type=str, default='200,50,20')
    parser.add_argument('--graph_hid_dims', type=str, default='200,50,20')
    # parser.add_argument('--save_path', type=str, default='checkpoint/triplet/')

    # train
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='checkpoint/triplet/')
    parser.add_argument('--batch_size', type=int, default=2048)

    # test
    parser.add_argument('--test_save_path', type=str, default='checkpoint/triplet/best_f1.pth')
    # parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pred', type=str, default='results/predict.csv')
    parser.add_argument('--emb1', type=str, default='results/embed_ligand.csv')
    parser.add_argument('--emb2', type=str, default='results/embed_receptor.csv')
    parser.add_argument('--threshold', type=float, default=0.5)

    # seed
    parser.add_argument('--seed', type=int, default=10)

    # cuda
    parser.add_argument('--use_cuda', type=str, default='cuda:0')

    # yml name
    parser.add_argument('--ymlname', type=str, default='configure_gen.yml')

    opt = parser.parse_args()

    yml = open(opt.ymlname, 'w')
    yml.write('DATASET:\n')
    yml.write('  NAME: %s\n'%(opt.dataname))
    yml.write('  TRAIN_ROOT: %s\n'%(opt.trainroot))
    yml.write('  TEST_ROOT: %s\n'%(opt.testroot))
    yml.write('  PRED_ROOT: %s\n'%(opt.predroot))
    yml.write('  MATRIX_ROOT: %s\n'%(opt.matrixroot))
    yml.write('  ADJ_ROOT: %s\n'%(opt.adjroot))

    yml.write('MODEL:\n')
    yml.write('  NAME: %s\n'%(opt.modelname))
    yml.write('  INPUT_DIM: %d\n'%(opt.input_dim))
    yml.write('  GRAPH_DIM: %d\n'%(opt.input_dim))

    mlp_hid_dims = opt.mlp_hid_dims.split(',')
    mlp_hid_dims = [int(x) for x in mlp_hid_dims]
    yml.write('  MLP_HID_DIMS: [%d'%(mlp_hid_dims[0]))
    for d in mlp_hid_dims[1:]:
        yml.write(',%d'%(d))
    yml.write(']\n')

    graph_hid_dims = opt.graph_hid_dims.split(',')
    graph_hid_dims = [int(x) for x in graph_hid_dims]
    yml.write('  GRAPH_HID_DIMS: [%d'%(graph_hid_dims[0]))
    for d in graph_hid_dims[1:]:
        yml.write(',%d'%(d))
    yml.write(']\n')
    yml.write('  SAVE_PATH: %s\n'%(opt.save_path))

    yml.write('TRAIN:\n')
    yml.write('  LR: %f\n'%(opt.lr))
    yml.write('  EPOCHS: %d\n'%(opt.epochs))
    yml.write('  SAVE_PATH: %s\n'%(opt.save_path))
    yml.write('  BATCH_SIZE: %d\n'%(opt.batch_size))

    yml.write('TEST:\n')
    yml.write('  SAVE_PATH: %s\n'%(opt.test_save_path))
    yml.write('  BATCH_SIZE: %d\n'%(opt.batch_size))
    yml.write('  PRED: %s\n'%(opt.pred))
    yml.write('  EMB1: %s\n'%(opt.emb1))
    yml.write('  EMB2: %s\n'%(opt.emb2))
    yml.write('  THRESHOLD: %f\n'%(opt.threshold))

    yml.write('SEED: %d\n'%(opt.seed))
    yml.write('SEED: %s\n'%(opt.use_cuda))


if __name__ == '__main__':
    main()