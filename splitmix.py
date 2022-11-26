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
            out = net(x)
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

# [atom_model, ... , atom_model] -> len(num_base)
global_model = model(cfg, cfg['global_width_idx'], cfg['model_width_list'])
atom_parameters_list = global_model.get_all_state_dict()
local_models = []
local_parameters_dict = dict()

agg_weight = []
for i in range(cfg['client_num']):
    agg_weight.append(len(net_dataidx_map[i])/len(X_train))               ### set to 1 / num
    # [atom_models(len:num_ens), ... , atom_models(len:num_ens)] -> len(num_users)
    local_models.append(model(cfg, cfg['model_width_idx'][i], cfg['model_width_list']))
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
num_base = int(1. / cfg['atom_width'])
for round in range(cfg["comm_round"]):
    user_idx = party_list_rounds[round]
    local_parameters, slim_shifts_per_usr = federation.distribute(user_idx, cfg, global_model)
    for idx, i in enumerate(user_idx):
        acc = local_train(round, local_models[i], local_parameters[idx], train_local_dls[i], test_dl, cfg)
        print("Round %d, client %d, width %.3f, local_acc: %f" % (round, i, cfg['model_width_list'][cfg['model_width_idx'][i]], acc))
        for idx1, param1 in enumerate(local_parameters[idx]):
                local_parameters[idx][idx1] = copy.deepcopy(local_models[i]._modules['atom_models'][idx1].state_dict())

    global_parameters = global_model.get_all_state_dict()
    global_parameters = federation.combine(global_parameters, local_parameters, user_idx, slim_shifts_per_usr, agg_weight, num_base)
    global_model.load_all_state_dict(global_parameters)
    global_model.cuda()
    # Esitimate BN statics
    # with torch.no_grad():
    #     global_model.train()
    #     for i, (x, target) in enumerate(train_dl_global):
    #         # for width_idx in range(cfg['global_width_idx']+1):
    #         global_model(x.cuda(), cfg['global_width_idx'])
    update_result = False
    # new_result = []
    # for width_idx in range(cfg['global_width_idx']+1):
    acc = compute_acc(global_model, test_dl)
    # new_result.append(acc)
    if best_acc < acc:
        best_acc = acc
        update_result = True
    print("Round %d, Global Accuracy: %f" % (round, acc))
    if update_result:
        best_result = best_acc
        print("Round %d, New Best Global Accuracy!" % (round))
    else:
        print("History best:",best_acc)
    global_model.to('cpu')
print(cfg)

