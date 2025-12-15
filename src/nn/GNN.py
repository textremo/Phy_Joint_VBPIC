import torch
import torch.nn as nn
try:
    from ..nn.FlexGRU import FlexGRU
except:
    from FlexGRU import FlexGRU
    
    
class TelecommGNN(nn.Module):
    '''
    @iter_num:      the iteration number
    @feat_n_num:    node feature number
    @nn_num:        the neuron number
    '''
    def __init__(self, iter_num, feat_n_numm, nn_num=64):
        super().__init__()
        self.iter_num = iter_num
        self.nn_num = nn_num
        