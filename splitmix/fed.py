import copy
import torch
import numpy as np
from collections import OrderedDict
import time

#TODO complete the distribution and aggeregation part
class Federation:
    def __init__(self, atom_model_params, agg_weight, cfg):
        self.atom_model_params = atom_model_params
        self.num_ens_list = [cfg['model_width_list'][idx] / cfg['atom_width']  for idx in cfg['model_width_idx']]
        self.agg_weight = agg_weight # weight used in fedavg

    def distribute(self, user_idxs):
        model_params = []
        for user_idx in user_idxs:
            num_ens = self.num_ens_list[user_idx]
            model_params.append([copy.deepcopy(self.atom_model_params[i]) for i in range(num_ens)])
        return model_params
    
    def combine(self, local_parameters, user_idxs):
        # TODO 
        print("combine logits")

        