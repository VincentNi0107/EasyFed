import random
import copy
import time

import numpy as np
import torch
from hetro.config import get_args
from hetro.resnet import resnet18
from hetro.fed import Federation
from utils import *

def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    return correct / float(total)

def local_train(net, para, train_data_loader, test_data_loader, cfg):
    net.load_state_dict(para)
    net.cuda()
    net.train()
    if cfg["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg["lr"], weight_decay=cfg["reg"])
    elif cfg["optimizer"] == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg["lr"], weight_decay=cfg["reg"],
                               amsgrad=True)
    elif cfg["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg["lr"], momentum=0.9,
                              weight_decay=cfg["reg"])
    criterion = nn.CrossEntropyLoss().cuda()
    test_acc = compute_acc(net, test_data_loader) 
    for epoch in range(cfg["epochs"]):
        for batch_idx, (x, target) in enumerate(train_data_loader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
    # test_acc = compute_acc(net, test_data_loader)        
    net.to('cpu')
    return test_acc



args, cfg = get_args()
print(cfg)
X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, cfg['client_num'], beta=args.beta)

n_party_per_round = int(cfg['client_num'] * args.sample_fraction)
party_list = [i for i in range(cfg['client_num'])]
party_list_rounds = []
if n_party_per_round != cfg['client_num']:
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
train_local_dls = []

global_model = resnet18(cfg, cfg['global_model_rate'])

global_parameters = global_model.state_dict()
federation = Federation(global_parameters, cfg)
local_models = []
for i in range(cfg['client_num']):
    local_models.append(resnet18(cfg, cfg['model_rate'][i]))
    dataidxs = net_dataidx_map[i]
    train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
    train_local_dls.append(train_dl_local)
    
for round in range(cfg["comm_round"]):
    user_idx = party_list_rounds[round]
    local_parameters, param_idx = federation.distribute(user_idx)
    for i in user_idx:
        acc = local_train(local_models[i], local_parameters[i], train_local_dls[i], test_dl, cfg)
        print("Round %d, client %d, width %.2f, local_acc: %f" % (round, i, cfg['model_rate'][i], acc))
        local_parameters[i] = copy.deepcopy(local_models[i].state_dict())
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    
    with torch.no_grad():
        test_model = resnet18(cfg, cfg['global_model_rate'], track=True).cuda()
        test_model.load_state_dict(global_model.state_dict(), strict=False)
        test_model.train()
        for i, (x, target) in enumerate(train_dl_global):
            test_model(x.cuda())
    acc = compute_acc(test_model, test_dl)
    print("Round %d, Global Accuracy: %f" % (round, acc))