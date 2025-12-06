import torch
import torch.nn as nn
try:
    from ..nn.FlexGRU import FlexGRU
except:
    from FlexGRU import FlexGRU
    
    
class TelecommGNN(nn.Module):
    '''
    @iter_num:  the iteration number
    @nf_num:    node feature number
    @neuron_num: the neuron
    '''
    def __init__(self, iter_num, neuron_num=64):
        super().__init__()