import copy
import torch
import numpy as np
from collections import OrderedDict
import time

#TODO complete the distribution and aggeregation part
class Federation:
    def __init__(self, atom_model_params, agg_weight, cfg): # self, global base model weight list (len:8), aggregated weight of each usr (len:num_users)
        self.atom_model_params = atom_model_params
        self.num_ens_list = [int(cfg['model_width_list'][idx] / cfg['atom_width'])  for idx in cfg['model_width_idx']]
        self.agg_weight = agg_weight # weight used in fedavg
        self.num_base = int(1. / cfg['atom_width'])
        self.user_base_sampler = shuffle_sampler(list(range(self.num_base)))

    def distribute(self, user_idxs, cfg, global_model):
        model_params = []
        slim_shifts_per_usr = []
        self.atom_model_params = copy.deepcopy(global_model.get_all_state_dict())
        for user_idx in user_idxs:
            num_ens = self.num_ens_list[user_idx]
            slim_shifts = self.sample_bases(num_ens, cfg)
            # TODO choose base model idx
            model_params.append([copy.deepcopy(self.atom_model_params[i]) for i in slim_shifts])
            slim_shifts_per_usr.append(slim_shifts)
        return model_params, slim_shifts_per_usr
    
    def combine(self, global_parameters, local_parameters, user_idx, slim_shifts_per_usr, agg_weight, num_base):
        for base_i in range(num_base):
            count_weight = 0        ### e.g total 0.3, -> weight 0.1/0.3 & 0.2/0.3
            for idx, slim_shifts in enumerate(slim_shifts_per_usr):
                if base_i in slim_shifts:
                    count_weight += weight
    
            for idx, slim_shifts in enumerate(slim_shifts_per_usr):
                if base_i in slim_shifts:
                    user_id = user_idx[idx]
                    weight = agg_weight[user_id]
                    # count_weight += weight
                    for key in global_parameters[base_i]:
                        old_tensor = global_parameters[base_i][key].data
                        new_tensor = local_parameters[idx][np.where(base_i == np.array(slim_shifts))[0][0]][key].data
                        if 'num_batches_tracked' in key:
                            # num_batches_tracked is a non trainable LongTensor and
                            # num_batches_tracked are the same for all clients for the given datasets
                            old_tensor.copy_(new_tensor)
                        else:
                            temp = (weight / count_weight) * new_tensor
                            old_tensor.add_(temp)
            # for key in global_parameters[base_i]:
                # if 'num_batches_tracked' not in key:
                    # global_parameters[base_i][key].data *= 1. / count_weight        ### divide as a whole?
        return global_parameters

    def sample_bases(self, num_ens, cfg):
        """Sample base models for the client.
        """
        # (Alg 2) Sample base models defined by shift index.
        slim_shifts = [self.user_base_sampler.next()]
        if num_ens > 1:
            _sampler = shuffle_sampler([v for v in self.user_base_sampler.arr if v != slim_shifts[0]])
            slim_shifts += [_sampler.next() for _ in range(num_ens - 1)]
        slim_ratios = [cfg['atom_width']] * num_ens
        print(f"slim_ratios={slim_ratios}, slim_shifts={slim_shifts}")
        return slim_shifts

class _Sampler(object):
    def __init__(self, arr):
        self.arr = copy.deepcopy(arr)

    def next(self):
        raise NotImplementedError()


class shuffle_sampler(_Sampler):
    def __init__(self, arr, rng=None):
        super().__init__(arr)
        if rng is None:
            rng = np.random
        rng.shuffle(self.arr)
        self._idx = 0
        self._max_idx = len(self.arr)

    def next(self):
        if self._idx >= self._max_idx:
            np.random.shuffle(self.arr)
            self._idx = 0
        v = self.arr[self._idx]
        self._idx += 1
        return v
