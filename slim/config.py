import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=12, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--lam_fm', type=float, default=1.0)
    parser.add_argument('--start', type=int, default=20, help='the server momentum (FedAvgM)')

    args = parser.parse_args()
    cfg = dict()
    cfg["comm_round"] = args.comm_round
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    cfg["reg"] = args.reg
    cfg["epochs"] = args.epochs
    cfg["self_dist"] = False
    cfg["recc_dist"] = False
    cfg["nova"] = False
    cfg["feature_match"] = True
    cfg["fm_idx"] = 0
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        cfg['classes_size'] = 10
        cfg['data_shape'] = [3, 32, 32]
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
        cfg['data_shape'] = [3, 32, 32]
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
        args.datadir = '/GPFS/data/ruiye/fssl/dataset/tiny_imagenet/tiny-imagenet-200'

    cfg['hidden_size'] = [64, 128, 256, 512]
    # cfg['hidden_size'] = [16, 32, 64, 128]
    cfg['model_width_list'] = [0.125,0.25,0.5,1.0]
    # cfg['model_width_list'] = [0.25,0.5,0.75,1.0]
    # cfg['model_width_idx'] = [2]
    # cfg['model_width_list'] = [0.2,0.4,0.6,0.8,1.0]
    # cfg['model_width_idx'] = [0,0,0,1,1,1,2,2,2,3,3,3]
    # cfg['model_width_idx'] = [0,0,0,1,1,1,2,2,2,2,2,2]
    # cfg['model_width_idx'] = [0,0,0,1,1,1,1,1,1,1,1,1]
    # cfg['model_width_idx'] = [3,3,3,3,3,3,3,3,3,3,3,3]
    # cfg['model_width_idx'] = [0,0,0,0,0,0]
    # cfg['model_width_idx'] = [2,2,2,2,2,2,2,2,2,2,2,2]
    # cfg['model_width_idx'] = [1,1,1,1,1,1,1,1,1,1,1,1]
    # cfg['model_width_list'] = [1.0]
    cfg['model_width_idx'] = [0,0,0,0,0,0,0,0,0,0,0,0]
    cfg['global_width_idx'] = max(cfg['model_width_idx'])
    cfg['client_num'] = len(cfg['model_width_idx'])
    cfg["partition"] = args.partition
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    return args , cfg