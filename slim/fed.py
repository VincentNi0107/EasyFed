import copy
import torch
import numpy as np
from collections import OrderedDict
import time

class Federation:
    def __init__(self, global_parameters, local_parameters_dict, cfg):
        self.global_parameters = global_parameters
        self.width_list = cfg['model_width_list']
        self.width_idx_list = cfg['model_width_idx']
        self.global_width_idx = cfg['global_width_idx']
        self.global_rate = self.width_list[cfg['global_width_idx']]
        self.local_parameters_dict = local_parameters_dict

    def distribute(self, user_idx):
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        width_rate_list = [self.width_list[self.width_idx_list[idx]] / self.global_rate for idx in user_idx]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'bn' in k:
                    if k in self.local_parameters_dict[self.width_idx_list[user_idx[m]]].keys():
                        local_parameters[m][k] = copy.deepcopy(v)
                elif 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'linear' in k:
                                local_parameters[m][k] = copy.deepcopy(v[:,:int(np.ceil(input_size * width_rate_list[m]))])
                            elif input_size == 3:
                                local_parameters[m][k] = copy.deepcopy(v[:int(np.ceil(output_size * width_rate_list[m]))])  
                            else:
                                local_parameters[m][k] = copy.deepcopy(v[:int(np.ceil(output_size * width_rate_list[m])), :int(np.ceil(input_size * width_rate_list[m]))])  
                        else:
                            raise ValueError('Invalid weight during model split')
                    else:
                        if 'linear' in k:
                            local_parameters[m][k] = copy.deepcopy(v)                        
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[:int(np.ceil(v.size(0) * width_rate_list[m]))])
                else:
                    raise ValueError('Invalid parameter during model split')
        return local_parameters

    def combine(self, local_parameters, user_idx):
        count = OrderedDict()
        width_rate_list = [self.width_list[self.width_idx_list[idx]] / self.global_rate for idx in user_idx]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if 'bn' in k:
                    if k in self.local_parameters_dict[self.width_idx_list[user_idx[m]]].keys():
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                elif 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'linear' in k:
                                tmp_v[:, :int(np.ceil(input_size * width_rate_list[m]))] += local_parameters[m][k]
                                count[k][:, :int(np.ceil(input_size * width_rate_list[m]))] += 1
                            elif input_size == 3:
                                tmp_v[:int(np.ceil(output_size * width_rate_list[m]))] += local_parameters[m][k]
                                count[k][:int(np.ceil(output_size * width_rate_list[m]))] += 1
                            else:
                                tmp_v[:int(np.ceil(output_size * width_rate_list[m])), :int(np.ceil(input_size * width_rate_list[m]))] += local_parameters[m][k]
                                count[k][:int(np.ceil(output_size * width_rate_list[m])), :int(np.ceil(input_size * width_rate_list[m]))] += 1
                        else:
                            raise ValueError('Invalid weight during model split')
                    else:
                        if 'linear' in k:
                            tmp_v += local_parameters[m][k]
                            count[k] += 1
                        else:
                            tmp_v[:int(np.ceil(v.size(0) * width_rate_list[m]))] += local_parameters[m][k]
                            count[k][:int(np.ceil(v.size(0) * width_rate_list[m]))] += 1
                else:
                    raise ValueError('Invalid parameter during model split')
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        return
    
    #TODO
    def nova_combine(self, local_parameters, user_idx):
        count = OrderedDict()
        # width_rate_list = [self.width_list[self.width_idx_list[idx]] / self.global_rate for idx in user_idx]
        width_rate_list = [0] + [ width / self.global_rate for width in self.width_list[:self.global_width_idx + 1]]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                user_width_idx = self.width_idx_list[user_idx[m]]
                if 'bn' in k:
                    if k in self.local_parameters_dict[self.width_idx_list[user_idx[m]]].keys():
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                elif 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            input_size_list = [int(np.ceil(input_size * width_rate)) for width_rate in width_rate_list[:user_width_idx + 2]]
                            output_size_list = [int(np.ceil(input_size * width_rate)) for width_rate in width_rate_list[:user_width_idx + 2]]
                            if 'linear' in k:
                                for width_idx in range(user_width_idx + 1):
                                    tmp_v[:, input_size_list[width_idx]:input_size_list[width_idx + 1]] += local_parameters[m][k][:, input_size_list[width_idx]:input_size_list[width_idx + 1]] / (user_width_idx + 1 - width_idx)
                                # tmp_v[:, :int(np.ceil(input_size * width_rate_list[m]))] += local_parameters[m][k]
                                count[k][:, :input_size_list[-1]] += 1
                            elif input_size == 3:
                                for width_idx in range(user_width_idx + 1):
                                    tmp_v[output_size_list[width_idx]:output_size_list[width_idx + 1]] += local_parameters[m][k][output_size_list[width_idx]:output_size_list[width_idx + 1]] / (user_width_idx + 1 - width_idx)
                                # tmp_v[:int(np.ceil(output_size * width_rate_list[m]))] += local_parameters[m][k]
                                count[k][:output_size_list[-1]] += 1
                            else:
                                for width_idx in reversed(range(user_width_idx + 1)):
                                    tmp_v[:output_size_list[width_idx + 1],:input_size_list[width_idx + 1]] = local_parameters[m][k][:output_size_list[width_idx + 1],:input_size_list[width_idx + 1]] / (user_width_idx + 1 - width_idx)
                                # tmp_v[:int(np.ceil(output_size * width_rate_list[m])), :int(np.ceil(input_size * width_rate_list[m]))] += local_parameters[m][k]
                                count[k][:output_size_list[-1], :input_size_list[-1]] += 1
                        else:
                            raise ValueError('Invalid weight during model split')
                    else:
                        if 'linear' in k:
                            tmp_v += local_parameters[m][k] / (user_width_idx + 1)
                            count[k] += 1
                        else:
                            output_size_list = [int(np.ceil(v.size(0) * width_rate)) for width_rate in width_rate_list[:user_width_idx + 2]]
                            for width_idx in range(user_width_idx + 1):
                                tmp_v[output_size_list[width_idx]:output_size_list[width_idx + 1]] += local_parameters[m][k][output_size_list[width_idx]:output_size_list[width_idx + 1]] / (user_width_idx + 1 - width_idx)
                            count[k][output_size_list[-1]] += 1
                else:
                    raise ValueError('Invalid parameter during model split')
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] += tmp_v[count[k] > 0].to(v.dtype)
        return