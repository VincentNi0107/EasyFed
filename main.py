from decimal import Rounded
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import pickle

from feature_matching.loss_f import *
from feature_matching.utils import *
from feddyn import *
from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
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
    parser.add_argument('--fm',action='store_true')
    parser.add_argument('--low_rate', type=float, default=0.0, help='low comm rate')
    parser.add_argument('--start_round', type=int, default=0, help='starting round')
    parser.add_argument('--shallow_freq', type=int, default=4, help='shallow_freq')
    parser.add_argument('--cls_only', type=int, default=0, help='shallow_freq')
    parser.add_argument('--time_zone', type=int, default=5, help='shallow_freq')
    parser.add_argument('--ref_type', type=str, default='all')
    parser.add_argument('--scale',action='store_true')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, round, device="cpu",global_centriods=None):
    net.cuda()
    # logger.info('Training network %s' % str(net_id))
    # logger.info('n_training: %d' % len(train_dataloader))
    # logger.info('n_test: %d' % len(test_dataloader))

    # train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    # test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,matchingr,out = net(x)
            loss = criterion(out, target)
            # if args.fm and round>=args.start:
            #     loss3=args.lam_fm*matching_cross_entropy(matchingr,target,global_centriods,0.1)
            #     loss+=loss3
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        # print('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch == epochs-1:
        #     train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    # logger.info(' ** Training complete **')
    print("test_acc:",test_acc)
    return test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args, round,
                      device="cpu",global_centriods=None):
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    # logger.info('Training network %s' % str(net_id))
    # logger.info('n_training: %d' % len(train_dataloader))
    # logger.info('n_test: %d' % len(test_dataloader))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,matchingr,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            if args.fm and round>=args.start:
                loss3=args.lam_fm*matching_cross_entropy(matchingr,target,global_centriods,0.1)
                loss+=loss3
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print('>> Training accuracy: %f' % train_acc)
    print('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    print(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu",global_centriods=None):
    # net = nn.DataParallel(net)
    net.cuda()
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    # test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # print('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # print('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_loss3_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            matchingr, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            
            loss = loss1 + loss2
            if args.fm and round>=args.start:
                loss3=args.lam_fm*matching_cross_entropy(matchingr,target,global_centriods,0.1)
                loss+=loss3
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            if args.fm and round>=args.start:
                epoch_loss3_collector.append(loss3.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        if args.fm and round>=args.start:
            epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
        else:
            epoch_loss3=0
        print('Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3:%f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2,epoch_loss3))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print('>> Training accuracy: %f' % train_acc)
    print('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    print(' ** Training complete **')
    return train_acc, test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu",global_centriods=None):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in enumerate(nets.values()):
        dataidxs = net_dataidx_map[net_id]

        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local=train_dl[net_id]
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        if args.alg == 'fedavg' or args.alg == 'fedregrad' or args.alg=='MIFA':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, round,
                                        device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, round, device,global_centriods)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device,global_centriods)
        elif args.alg == 'local_training':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    

    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    train_local_dls=[]    
    val_local_dls=[]
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        dataidxs_t = dataidxs[:int(0.8*len(dataidxs))]
        dataidxs_v = dataidxs[int(0.8*len(dataidxs)):]
        train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_t)
        train_local_dls.append(train_dl_local)
        val_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, 200, 32, dataidxs_v)
        val_local_dls.append(val_dl_local)
        
    # for i in range(len(para_list)):
        # print(para_list[i],'\t',global_model.state_dict()[para_list[i]].size())
    
    best_acc=0
    best_test_acc=0
    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                        
        for round in range(n_comm_rounds):
            print("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            
            global_model.eval()
            
            
            # if (round+1)%10==0:
            #     global_model.cuda()
            #     for idx in range(len(val_local_dls)):
            #         for batch_idx, (x, target) in enumerate(val_local_dls[idx]):
            #             x, target = x.cuda(), target.cuda()
            #             target = target.long()
            #             x.requires_grad = False
            #             target.requires_grad = False
            #             feature1, feature2, _ = global_model(x)
            #             if idx==0:
            #                 features1=feature1
            #                 features2=feature2
            #                 labels=target
            #             else:
            #                 features1=torch.cat((features1,feature1),0)
            #                 features2=torch.cat((features2,feature2),0)
            #                 labels=torch.cat((labels,target),0)
            #             break
            #     global_model.to('cpu')
            #     np.save("./feature/feature1_%d.npy"%(round),features1.cpu().detach().numpy())
            #     np.save("./feature/feature2_%d.npy"%(round),features2.cpu().detach().numpy())
            #     np.save("./feature/label_%d.npy"%(round),labels.cpu().detach().numpy())
            
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)



            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl))
            global_model.cuda()
            #feature_matching
            # global_centroids = global_centroids.cpu()
            # local_centroids, local_distributions = get_client_centroids_info(global_model, dataloaders=train_local_dls, model_name=args.model, dataset_name=args.dataset)
            # global_centroids = get_global_centroids(local_centroids, local_distributions, global_centroids, momentum=0.0)
            # global_centroids = global_centroids.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')

    elif args.alg=='fedregrad':
        test_acc_list=[]
        lastParticipateround=[]
        allOnes=[]
        deltPool=dict()
        for i in range(args.n_parties):
            lastParticipateround.append(-1)
            allOnes.append(i)
        periodPartRound=[]
        periodPartRound.append(allOnes)
        for i in range(len(party_list_rounds)):
            periodPartRound.append(party_list_rounds[int(i/args.time_zone)])
        
        for round in range(n_comm_rounds):
            print("round:",round)
            party_list_this_round = periodPartRound[round]
            print("participate client idx",party_list_this_round)
            for partClient in party_list_this_round:
                lastParticipateround[partClient]=round
                
            global_w = global_model.state_dict()
            old_w = copy.deepcopy(global_model.state_dict())
            
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            traindlThisRound=[train_local_dls[k] for k in party_list_this_round]
            valdlThisRound=[val_local_dls[k] for k in party_list_this_round]
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=traindlThisRound, test_dl=test_dl,round=round, device=device)
            print("finish local training")
            
            delta_w_this_round=dict()
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
            if round%args.time_zone==0 or round == 0:
                for net_id,net in nets_this_round.items():
                    delt=copy.deepcopy(global_w)
                    net_para = net.state_dict()
                    for key in delt:
                        delt[key] = net_para[key] - old_w[key]
                    delta_w_this_round[net_id]=delt
                deltPool[round]=delta_w_this_round
            
            for net_id, net in nets_this_round.items():
                net_para = net.state_dict()
                if net_id == party_list_this_round[0]:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            ref_last=copy.deepcopy(global_w)
            ref_new=copy.deepcopy(global_w)
            for i in range(args.n_parties):
                if i not in party_list_this_round:
                    commonClient=[]
                    lastRound=lastParticipateround[i]
                    if lastRound==-1:
                        continue               
                    lastRoundPart=periodPartRound[lastRound]
                    RoundPart=periodPartRound[round]
                    last_w_i=deltPool[lastRound][i]
                    for client in lastRoundPart:
                        if client in RoundPart and client!=i:
                            commonClient.append(client)                    
                    if args.ref_type=='all':
                        for idx in range(len(commonClient)):
                            netpara=deltPool[lastRound][commonClient[idx]]
                            netparanew=nets_this_round[commonClient[idx]].state_dict()
                            if idx==0:
                                for key in netpara:
                                    ref_last[key]=netpara[key]/len(commonClient)
                                    ref_new[key]=netparanew[key]/len(commonClient)
                            else:
                                for key in netpara:
                                    ref_last[key]+=netpara[key]/len(commonClient)
                                    ref_new[key]+=netparanew[key]/len(commonClient)
                        print("client ",i," off at round ",round,"ref client ",commonClient,"at round ",lastRound)

                    elif args.ref_type=='one':
                        for idx in range(len(commonClient)):
                            last_w_ref=deltPool[lastRound][commonClient[idx]]
                            diff=0
                            for key in last_w_ref:
                                diff+=torch.abs(last_w_ref[key]-last_w_i[key]).sum().sum()
                            if idx==0:
                                mindiff=diff
                                minidx=idx
                            elif diff<mindiff:
                                mindiff=diff
                                minidx=idx
                        ref_new=nets_this_round[commonClient[minidx]].state_dict()      
                        ref_last=deltPool[lastRound][commonClient[minidx]]
                        print("client ",i," off at round ",round,"ref client ",commonClient[minidx],"at round ",lastRound)
                    scale=1
                    if args.scale: 
                        lastnorm=0
                        newnorm=0
                        for key in ref_last:
                            newnorm+=torch.abs(ref_new[key]-old_w[key]).sum().sum()
                            lastnorm+=torch.abs(ref_last[key]).sum().sum()
                        scale=newnorm/lastnorm
                    print("scale",scale)
                    for key in ref_last:
                        global_w[key] += (ref_new[key]+(last_w_i[key]-ref_last[key])*scale)* fed_avg_freqs[i]
                        
        
            global_model.load_state_dict(global_w)
            global_model.cuda()
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                print('>> Global Model Val accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
            test_acc_list.append(test_acc)
            global_model.to('cpu')                
        print(test_acc_list)
                        
    elif args.alg=='MIFA':
        lastParticipateround=[]
        allOnes=[]
        deltPool=dict()
        for i in range(args.n_parties):
            lastParticipateround.append(-1)
            allOnes.append(i)
        periodPartRound=[]
        periodPartRound.append(allOnes)
        for i in range(len(party_list_rounds)):
            periodPartRound.append(party_list_rounds[int(i/args.time_zone)])
        
        for round in range(n_comm_rounds):
            print("round:",round)
            party_list_this_round = periodPartRound[round]
            print("participate client idx",party_list_this_round)
            for partClient in party_list_this_round:
                lastParticipateround[partClient]=round
                
            global_w = global_model.state_dict()
            old_w = copy.deepcopy(global_model.state_dict())
            
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            traindlThisRound=[train_local_dls[k] for k in party_list_this_round]
            valdlThisRound=[val_local_dls[k] for k in party_list_this_round]
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=traindlThisRound, test_dl=test_dl,round=round, device=device)
            print("finish local training")
            
            delta_w_this_round=dict()
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
            if round%args.time_zone==0 or round == 0:
                for net_id,net in nets_this_round.items():
                    delt=copy.deepcopy(global_w)
                    net_para = net.state_dict()
                    for key in delt:
                        delt[key] = net_para[key] - old_w[key]
                    delta_w_this_round[net_id]=delt
                deltPool[round]=delta_w_this_round
            
            for net_id, net in nets_this_round.items():
                net_para = net.state_dict()
                if net_id == party_list_this_round[0]:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            ref_last=copy.deepcopy(global_w)
            ref_new=copy.deepcopy(global_w)
            for i in range(args.n_parties):
                if i not in party_list_this_round:
                    commonClient=[]
                    lastRound=lastParticipateround[i]
                    if lastRound==-1:
                        continue
                    lastRoundPart=periodPartRound[lastRound]
                    RoundPart=periodPartRound[round]
                    last_w_i=deltPool[lastRound][i]

                    for key in last_w_i:
                        # scale=torch.div(ref_new[key],ref_last[key])
                        # global_w[key] += (old_w[key]+ref_new[key]+torch.mul(last_w_i[key] - ref_last[key],scale))* fed_avg_freqs[i]
                        global_w[key] += (old_w[key]+last_w_i[key])* fed_avg_freqs[i]

        
            global_model.load_state_dict(global_w)
            global_model.cuda()
            val_acc, _ = compute_accuracy(global_model, valdlThisRound, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                print('>> Global Model Val accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
           
            global_model.to('cpu')                
                    
    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            print("round:",round,party_list_rounds[round])
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())
            
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            traindlThisRound=[train_local_dls[k] for k in party_list_this_round]
            valdlThisRound=[val_local_dls[k] for k in party_list_this_round]
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=traindlThisRound, test_dl=test_dl,round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        print(key,net_para[key].shape)
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            val_acc, _ = compute_accuracy(global_model, valdlThisRound, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            # test_result.append(test_acc)
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
           
            # mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')
            
    elif args.alg == 'feddyn':
        alpha_coef = 1e-2
        n_par = len(get_mdl_params([global_model])[0])
        local_param_list = np.zeros((args.n_parties, n_par)).astype('float32')
        init_par_list=get_mdl_params([global_model], n_par)[0]
        clnt_params_list  = np.ones(args.n_parties).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
        avg_model = copy.deepcopy(global_model)
        # avg_model.load_state_dict(copy.deepcopy(dict(global_model.named_parameters())))

        all_model = copy.deepcopy(global_model)
        # all_model.load_state_dict(copy.deepcopy(dict(global_model.named_parameters())))

        cld_model = copy.deepcopy(global_model)
        # cld_model.load_state_dict(copy.deepcopy(dict(global_model.named_parameters())))
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
        weight_list = np.asarray([len(net_dataidx_map[i]) for i in range(args.n_parties)])
        weight_list = weight_list / np.sum(weight_list) * args.n_parties
        for round in range(n_comm_rounds):
            print("round:",round)
            party_list_this_round = party_list_rounds[round]
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32).cuda()

            for clnt in party_list_this_round:
                print('---- Training client %d' %clnt)
                train_dataloader=train_local_dls[clnt]
                model = copy.deepcopy(global_model).cuda()
                # Warm start from current avg model
                model.load_state_dict(cld_model.state_dict())
                for params in model.parameters():
                    params.requires_grad = True

                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=device)
                loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=alpha_coef+args.reg)
                model.train()
                model.cuda()
                for e in range(args.epochs):
                    # Training
                    # epoch_loss_collector = []
                    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                        
                        batch_x.requires_grad=False
                        batch_y.requires_grad=False
                        
                        optimizer.zero_grad()
                        _,_,y_pred = model(batch_x)
                        
                        ## Get f_i estimate 
                        loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
                        loss_f_i = loss_f_i / list(batch_y.size())[0]
                        
                        # Get linear penalty on the current parameter estimates
                        local_par_list = None
                        for param in model.parameters():
                            if not isinstance(local_par_list, torch.Tensor):
                            # Initially nothing to concatenate
                                local_par_list = param.reshape(-1)
                            else:
                                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                        loss_algo = alpha_coef_adpt * torch.sum(local_par_list * (-cld_mdl_param_tensor + local_param_list_curr))
                        loss = loss_f_i + loss_algo

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10) # Clip gradients
                        optimizer.step()
                    # print("epoch:",e," loss:",loss.item())
                
                # Freeze model
                for params in model.parameters():
                    params.requires_grad = False
                model.eval()
                curr_model_par = get_mdl_params([model], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param = np.mean(clnt_params_list[party_list_this_round], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

            avg_model = set_client_from_params(copy.deepcopy(global_model), avg_mdl_param)
            all_model = set_client_from_params(copy.deepcopy(global_model), np.mean(clnt_params_list, axis = 0))
            cld_model = set_client_from_params(copy.deepcopy(global_model), cld_mdl_param) 
        
            
            avg_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            val_acc, _ = compute_accuracy(avg_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(avg_model, test_dl, get_confusion_matrix=True, device=device)
            # test_result.append(test_acc)
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
           
            # mkdirs(args.modeldir+'fedavg/')
            avg_model.to('cpu')
    
    elif args.alg == 'fedcc':
        test_result=[]
        para_list=list(global_model.state_dict().keys())
        if args.cls_only:
            deep_paralist=para_list[-2:]
            for layer in deep_paralist:
                print(layer)
        else:
            num_low=int(len(para_list)*args.low_rate)
            print("num of low",num_low)
            deep_paralist=para_list[num_low:]
        
        for round in range(n_comm_rounds):
            print("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())
            para_name=para_list
            
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            
            for net in nets_this_round.values():
                net_para = net.state_dict()
                for i in range(len(para_name)):
                    net_para[para_name[i]]=global_w[para_name[i]]
                net.load_state_dict(net_para)
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl,round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            # for net_id, net in enumerate(nets_this_round.values()):
            #     net_para = net.state_dict()
            #     if net_id == 0:
            #         for key in net_para:
            #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            #     else:
            #         for key in net_para:
            #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            if(round>args.start_round and round%args.shallow_freq!=0):
                para_name=deep_paralist
                print("comm deep only")
            else:
                print("comm all")
                para_name=para_list
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for i in range(len(para_name)):
                        global_w[para_name[i]] = net_para[para_name[i]] * fed_avg_freqs[net_id]
                else:
                    for i in range(len(para_name)):
                        global_w[para_name[i]] += net_para[para_name[i]] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl))
            global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_local_dls, device=device, multiloader=True)
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            test_result.append(test_acc)
            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
           
            # mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')
        with open(str(args.model)+"lowrate-"+str(args.low_rate)+"comfeq-"+str(args.shallow_freq)+"epo"+str(args.epochs),"wb") as fp:
            pickle.dump(test_result,fp)
        print(test_result)
        print("niid2lowrate-"+str(args.low_rate)+"comfeq-"+str(args.shallow_freq)+"epo"+str(args.epochs))
            
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            print("round:",round)
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls,test_dl=test_dl, global_model = global_model,round=round, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl))

            global_model.cuda()
            val_acc, _ = compute_accuracy(global_model, val_local_dls, device=device, multiloader=True)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            if(best_acc<val_acc):
                best_acc=val_acc
                best_test_acc=test_acc
                print('New Best val acc:%f , test acc:%f'%(val_acc,test_acc))
            else:
                # logger.info('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Train accuracy: %f' % val_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Best accuracy: %f' % best_test_acc)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')

