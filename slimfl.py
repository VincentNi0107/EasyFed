import random
import copy
import time

import numpy as np
import torch
from slim.config import get_args
from slim.resnet import resnet18,resnet34,resnet50
from slim.fed import Federation
from utils import *

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss
    
class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss
    
def compute_acc(net, width_idx, test_data_loader):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x, width_idx)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    return correct / float(total)

def local_train(round, net, width_idx, para, train_data_loader, test_data_loader, cfg):
    net.load_state_dict(para)
    net.cuda()
    if cfg["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg["lr"], weight_decay=cfg["reg"])
    elif cfg["optimizer"] == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg["lr"], weight_decay=cfg["reg"],
                               amsgrad=True)
    elif cfg["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg["lr"], momentum=0.9,
                              weight_decay=cfg["reg"])
    criterion = nn.CrossEntropyLoss()
    kd_loss = KL_Loss()
    # test_acc = []
    # for idx in range(width_idx+1):         
    #     test_acc.append(compute_acc(net, idx, test_data_loader)) 
    # test_acc = compute_acc(net, test_data_loader) 
    if cfg["knowledge_transfer"] and round > 2:
        logits_list = []
        net.eval()
        with torch.no_grad():
            for x, target in train_data_loader:
                teacher_logits = []
                for idx in range(width_idx + 1):
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    out = net(x, idx)
                    teacher_logits.append(out.detach())
                logits_list.append(torch.mean(torch.stack(teacher_logits, dim=-1), dim=-1).cpu())
    net.train()
    for epoch in range(cfg["epochs"]):
        for batch_idx, (x, target) in enumerate(train_data_loader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            target = target.long()
            if cfg["feature_kd"]:
                for idx in range(width_idx + 1):               
                    out, new_feature = net(x, idx, True)
                    if round > 2:
                        if idx <= cfg["fkd_idx"] :
                            feature = new_feature.detach()
                            loss = criterion(out, target)  
                        else:
                            loss = criterion(out, target) + nn.functional.mse_loss(new_feature[:,:feature.shape[1]], feature)
                            add_new_feature = new_feature.detach()
                            feature = torch.cat((feature, add_new_feature[:,feature.shape[1]:]),1)
                    else:
                        loss = criterion(out, target)
                    loss.backward()
                    net.clear_grad(idx)
                    optimizer.step()
            elif cfg["self_dist"]:   
                out = net(x, width_idx)
                loss = criterion(out, target)
                loss.backward()
                net.clear_grad(width_idx)
                optimizer.step()
                teacher_logits = out.detach()
                for idx in range(width_idx):               
                    out = net(x, idx)
                    loss = kd_loss(out, teacher_logits) + criterion(out, target)
                    loss.backward()
                    net.clear_grad(idx)
                    optimizer.step()
            elif cfg["knowledge_transfer"] and round > 2:
                teacher_logits = logits_list[batch_idx].cuda()
                for idx in range(width_idx + 1):               
                    out = net(x, idx)
                    loss = criterion(out, target) + kd_loss(out, teacher_logits)
                    teacher_logits.cpu()
                    loss.backward()
                    net.clear_grad(idx)
                    optimizer.step()
            elif cfg["fedslim"]:
                for idx in range(width_idx + 1):
                    out = net(x, idx)
                    loss = criterion(out, target)
                    loss.backward()
                optimizer.step()                
            else:
                for idx in range(width_idx + 1):
                    out = net(x, idx)
                    loss = criterion(out, target)
                    loss.backward()
                    net.clear_grad(idx)
                    optimizer.step()
                    
    test_acc = []
    for idx in range(width_idx+1):         
        test_acc.append(compute_acc(net, idx, test_data_loader))        
    net.to('cpu')
    return test_acc



args, cfg = get_args()

seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

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
if args.dataset == 'cifar10':
    model = resnet18
elif args.dataset == 'cifar100':
    model = resnet34
else:
    model = resnet50

global_model = model(cfg, cfg['global_width_idx'])
global_parameters = global_model.state_dict()
local_models = []
local_parameters_dict = dict()

agg_weight = []
for i in range(cfg['client_num']):
    agg_weight.append(len(net_dataidx_map[i])/len(X_train))
    local_models.append(model(cfg, cfg['model_width_idx'][i]))
    if cfg['model_width_idx'][i] not in local_parameters_dict:     
        local_parameters_dict[cfg['model_width_idx'][i]] = local_models[i].state_dict()
    dataidxs = net_dataidx_map[i]
    train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
    train_local_dls.append(train_dl_local)
    

# for k, v in local_parameters_dict.items():
#     print(k)
# time.sleep(1000)

federation = Federation(global_parameters, local_parameters_dict, agg_weight, cfg)
best_result = [0 for _ in range(cfg['global_width_idx']+1)]
best_acc = 0
print(cfg)

for round in range(cfg["comm_round"]):
    user_idx = party_list_rounds[round]
    local_parameters = federation.distribute(user_idx)
    for idx, i in enumerate(user_idx):
        acc_list = local_train(round, local_models[i], cfg['model_width_idx'][i], local_parameters[idx], train_local_dls[i], test_dl, cfg)
        for width_idx, acc in enumerate(acc_list):
            print("Round %d, client %d, max_width %.2f, width %.2f, local_acc: %f" % (round, i, cfg['model_width_list'][cfg['model_width_idx'][i]], cfg['model_width_list'][width_idx], acc))
        if cfg["nova"]:
            model_dict = local_models[i].state_dict()
            for key in model_dict:
                local_parameters[idx][key] = model_dict[key] - local_parameters[idx][key]
        else:
            local_parameters[idx] = copy.deepcopy(local_models[i].state_dict())

    # torch.save(local_parameters[0],"net1.pkl")
    # torch.save(local_parameters[1],"net2.pkl")
    # local_parameters[0] = torch.load("net1.pkl")
    # local_parameters[1] = torch.load("net2.pkl")
    # local_models[0].load_state_dict(local_parameters[0])
    # local_models[0].cuda()
    # acc = compute_acc(local_models[0], 0, test_dl)
    # print("local model acc: ",acc)
    # for key in global_parameters:
    #     global_parameters[key] = local_parameters[0][key] * 0.5 + local_parameters[1][key] * 0.5
    if cfg["nova"]:
        federation.nova_combine(local_parameters, user_idx)
    else:        
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
