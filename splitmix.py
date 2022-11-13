import random
import copy
import time

import numpy as np
import torch
from splitmix.config import get_args
from splitmix.ensemblenet import EnsembleNet
from splitmix.fed import Federation
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

def local_train(round, net, para, train_data_loader, test_data_loader, cfg):
    net.load_params(para)
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
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        for batch_idx, (x, target) in enumerate(train_data_loader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            target = target.long()        
            out = net(x, idx)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

    test_acc = compute_acc(net, test_data_loader)       
    net.to('cpu')
    return test_acc



args, cfg = get_args()
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

train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                            args.datadir,
                                                                            args.batch_size,
                                                                            32)
train_local_dls = []
model = EnsembleNet

global_model = model(cfg, cfg['global_width_idx'])
atom_parameters_list = global_model.get_all_state_dict()
local_models = []
local_parameters_dict = dict()

agg_weight = []
for i in range(cfg['client_num']):
    agg_weight.append(len(net_dataidx_map[i])/len(X_train))
    local_models.append(model(cfg, cfg['model_width_idx'][i]))
    # if cfg['model_width_idx'][i] not in local_parameters_dict:     
    #     local_parameters_dict[cfg['model_width_idx'][i]] = local_models[i].state_dict()
    dataidxs = net_dataidx_map[i]
    train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
    train_local_dls.append(train_dl_local)
    
# TODO: print and take a look at the name of model parameters, which is useful in model aggeregation and distribution
# for k, v in global_parameters.items():
#     print(k)


federation = Federation(atom_parameters_list, agg_weight, cfg)
best_result = [0 for _ in range(cfg['global_width_idx']+1)]
best_acc = 0
print(cfg)

for round in range(cfg["comm_round"]):
    user_idx = party_list_rounds[round]
    local_parameters = federation.distribute(user_idx)
    for idx, i in enumerate(user_idx):
        acc_list = local_train(round, local_models[i], local_parameters[idx], train_local_dls[i], test_dl, cfg)
        for width_idx, acc in enumerate(acc_list):
            print("Round %d, client %d, max_width %.2f, width %.2f, local_acc: %f" % (round, i, cfg['model_width_list'][cfg['model_width_idx'][i]], acc))
        local_parameters[idx] = copy.deepcopy(local_models[i].state_dict())

    federation.combine(local_parameters, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    global_model.cuda()
    # Esitimate BN statics
    # with torch.no_grad():
    #     global_model.train()
    #     for i, (x, target) in enumerate(train_dl_global):
    #         # for width_idx in range(cfg['global_width_idx']+1):
    #         global_model(x.cuda(), cfg['global_width_idx'])
    update_result = False
    new_result = []
    for width_idx in range(cfg['global_width_idx']+1):
        acc = compute_acc(global_model, width_idx, test_dl)
        new_result.append(acc)
        if best_acc < acc:
            best_acc = acc
            update_result = True
        print("Round %d, width %.2f, Global Accuracy: %f" % (round, cfg['model_width_list'][width_idx], acc))
    if update_result:
        best_result = new_result
        print("Round %d, New Best Global Accuracy!" % (round))
    else:
        print("History best:",best_result)
    global_model.to('cpu')
print(cfg)