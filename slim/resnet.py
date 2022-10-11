import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, scaler_rate_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.scaler_rate_list = scaler_rate_list

    def forward(self, input, width_idx):
        # input = input / self.scaler_rate_list[width_idx] if self.training else input
        y = self.bn[width_idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]

    def forward(self, input, width_idx):
        self.in_channels = self.in_channels_list[width_idx]
        self.out_channels = self.out_channels_list[width_idx]
        self.groups = self.groups_list[width_idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list

    def forward(self, input, width_idx):
        self.in_features = self.in_features_list[width_idx]
        self.out_features = self.out_features_list[width_idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)
     
class Block(nn.Module):
    expansion = 1

    def __init__(self, cfg, in_planes, planes, width_list, stride):
        super(Block, self).__init__()
        self.n1 = SwitchableBatchNorm2d([int(np.ceil(in_planes * wid)) for wid in width_list], [wid / cfg['model_width_list'][cfg['global_width_idx']] for wid in width_list])
        self.n2 = SwitchableBatchNorm2d([int(np.ceil(planes * wid)) for wid in width_list], [wid / cfg['model_width_list'][cfg['global_width_idx']] for wid in width_list])

        self.conv1 = SlimmableConv2d([int(np.ceil(in_planes * wid)) for wid in width_list], [int(np.ceil(planes * wid)) for wid in width_list], kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = SlimmableConv2d([int(np.ceil(planes * wid)) for wid in width_list], [int(np.ceil(planes * wid)) for wid in width_list], kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SlimmableConv2d([int(np.ceil(in_planes * wid)) for wid in width_list], [int(np.ceil(self.expansion * planes * wid)) for wid in width_list], kernel_size=1, stride=stride, bias=False)

    def forward(self, x, width_idx):
        out = F.relu(self.n1(x, width_idx))
        shortcut = self.shortcut(out, width_idx) if hasattr(self, 'shortcut') else x
        out = self.conv1(out, width_idx)
        out = self.conv2(F.relu(self.n2(out, width_idx)), width_idx)
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cfg, in_planes, planes, width_list, stride):
        super(Bottleneck, self).__init__()
        self.n1 = SwitchableBatchNorm2d([int(np.ceil(in_planes * wid)) for wid in width_list], [wid / cfg['model_width_list'][cfg['global_width_idx']] for wid in width_list])
        self.n2 = SwitchableBatchNorm2d([int(np.ceil(planes * wid)) for wid in width_list], [wid / cfg['model_width_list'][cfg['global_width_idx']] for wid in width_list])
        self.n3 = SwitchableBatchNorm2d([int(np.ceil(planes * wid)) for wid in width_list], [wid / cfg['model_width_list'][cfg['global_width_idx']] for wid in width_list])


        self.conv1 = SlimmableConv2d([int(np.ceil(in_planes * wid)) for wid in width_list], [int(np.ceil(planes * wid)) for wid in width_list], kernel_size=1, bias=False)
        self.conv2 = SlimmableConv2d([int(np.ceil(planes * wid)) for wid in width_list], [int(np.ceil(planes * wid)) for wid in width_list], kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = SlimmableConv2d([int(np.ceil(planes * wid)) for wid in width_list], [int(np.ceil(self.expansion * planes * wid)) for wid in width_list], kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SlimmableConv2d([int(np.ceil(in_planes * wid)) for wid in width_list], [int(np.ceil(self.expansion * planes * wid)) for wid in width_list], kernel_size=1, stride=stride, bias=False)

    def forward(self, x, width_idx):
        out = F.relu(self.n1(x, width_idx))
        shortcut = self.shortcut(out, width_idx) if hasattr(self, 'shortcut') else x
        out = self.conv1(out, width_idx)
        out = self.conv2(F.relu(self.n2(out, width_idx)), width_idx)
        out = self.conv3(F.relu(self.n3(out, width_idx)), width_idx)
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, cfg, hidden_size, block, num_blocks, num_classes, width_list):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.in_planes = hidden_size[0]
        self.conv1 = SlimmableConv2d([3 for _ in range(len(width_list))], [int(np.ceil(hidden_size[0] * wid)) for wid in width_list], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], width_list, stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], width_list, stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], width_list, stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], width_list, stride=2)
        self.n4 = SwitchableBatchNorm2d([int(np.ceil(hidden_size[3] * block.expansion * wid)) for wid in width_list], [wid / cfg['model_width_list'][cfg['global_width_idx']] for wid in width_list])
        
        self.linear = SlimmableLinear([int(np.ceil(hidden_size[3] * block.expansion * wid)) for wid in width_list], [num_classes for _ in range(len(width_list))])

    def _make_layer(self, block, planes, num_blocks, width_list, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.cfg, self.in_planes, planes, width_list, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, input, width_idx, return_feature = False):
        # x = input['img']
        out = self.conv1(input, width_idx)
        for layer in self.layer1:
            out = layer(out, width_idx)
        for layer in self.layer2:
            out = layer(out, width_idx)
        for layer in self.layer3:
            out = layer(out, width_idx)
        for layer in self.layer4:
            out = layer(out, width_idx)
        out = F.relu(self.n4(out, width_idx))
        out = F.adaptive_avg_pool2d(out, 1)
        feature = out.view(out.size(0), -1)
        logits = self.linear(feature, width_idx)
        if return_feature:
            return logits, feature
        return logits


def resnet18(cfg, width_idx):
    classes_size = cfg['classes_size']
    hidden_size = cfg['hidden_size']
    width_list = cfg['model_width_list'][:width_idx + 1]
    model = ResNet(cfg, hidden_size, Block, [2, 2, 2, 2], classes_size, width_list)
    model.apply(init_param)
    return model


def resnet34(cfg, width_idx):
    classes_size = cfg['classes_size']
    hidden_size = cfg['hidden_size']
    width_list = cfg['model_width_list'][:width_idx + 1]
    model = ResNet(cfg, hidden_size, Block, [3, 4, 6, 3], classes_size, width_list)
    model.apply(init_param)
    return model


def resnet50(cfg, width_idx):
    classes_size = cfg['classes_size']
    hidden_size = cfg['hidden_size']
    width_list = cfg['model_width_list'][:width_idx + 1]
    model = ResNet(cfg, hidden_size, Bottleneck, [3, 4, 6, 3], classes_size, width_list)
    model.apply(init_param)
    return model

#TODO
def resnet101(cfg, model_rate=1, track=False):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = ResNet(cfg, data_shape, hidden_size, Bottleneck, [3, 4, 23, 3], classes_size, scaler_rate, track)
    model.apply(init_param)
    return model

#TODO
def resnet152(cfg, model_rate=1, track=False):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = ResNet(cfg, data_shape, hidden_size, Bottleneck, [3, 8, 36, 3], classes_size, scaler_rate, track)
    model.apply(init_param)
    return model
