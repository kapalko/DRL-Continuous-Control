## derived from Udacity DDPG workbook

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    dim = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(dim)
    return (-lim, lim)