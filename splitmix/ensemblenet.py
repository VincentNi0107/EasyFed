import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, width, stride):
        super(Block, self).__init__()
        self.n1 = nn.BatchNorm2d(int(in_planes * width))
        self.n2 = nn.BatchNorm2d(int(planes * width))

        self.conv1 = nn.Conv2d(int(in_planes * width), int(planes * width), kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(int(planes * width), int(planes * width), kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(int(in_planes * width), int(self.expansion * planes * width), kernel_size=1, stride=stride, bias=False)

    def forward(self, x, width_idx):
        out = F.relu(self.n1(x, width_idx))
        shortcut = self.shortcut(out, width_idx) if hasattr(self, 'shortcut') else x
        out = self.conv1(out, width_idx)
        out = self.conv2(F.relu(self.n2(out, width_idx)), width_idx)
        out += shortcut
        return out
    
class ResNet(nn.Module):
    def __init__(self, hidden_size, block, num_blocks, num_classes, width):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(3, int(hidden_size[0] * width), kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], width, stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], width, stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], width, stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], width, stride=2)
        self.n4 = nn.BatchNorm2d(int(hidden_size[3] * block.expansion * width))
        
        self.linear = nn.Linear(int(hidden_size[3] * block.expansion * width), num_classes)

    def _make_layer(self, block, planes, num_blocks, width, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, width, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, input):
        out = self.conv1(input)
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        for layer in self.layer3:
            out = layer(out)
        for layer in self.layer4:
            out = layer(out)
        out = F.relu(self.n4(out))
        out = F.adaptive_avg_pool2d(out, 1)
        feature = out.view(out.size(0), -1)
        logits = self.linear(feature)
        return logits


class EnsembleNet(nn.Module):
    def __init__(self, cfg, width_idx):
        super(EnsembleNet, self).__init__()
        self.num_ens = int(cfg['model_width_list'][width_idx] / cfg['atom_width']) # 1.0/0.125 = 8
        self.atom_models = nn.ModuleList([ResNet(cfg['hidden_size'], Block, [2, 2, 2, 2], cfg['classes_size'], cfg['atom_width']) for _ in range(self.num_ens)])
        
        #TODO initialize models
        
        
    def forward(self, x):
        logits = [atom_model(x) for atom_model in self.atom_models]
        if len(logits) > 1:
            logits = torch.mean(torch.stack(logits, dim=-1), dim=-1)
        else:
            logits = logits[0]
        return logits
    
    def get_all_state_dict(self):
        model_state_dicts = [copy.deepcopy(atom_model.state_dict()) for atom_model in self.atom_models]
        return model_state_dicts
            
    def get_state_dict(self, idx):
        return self.atom_models[idx].state_dict()
    
    def load_params(self,  model_param_list):
        for i, atom_model in enumerate(self.atom_models):
            atom_model.load_state_dict(model_param_list[i])
