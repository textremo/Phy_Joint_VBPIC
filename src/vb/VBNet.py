import torch
import torch.nn as nn
from torch import arange, ones, zeros, eye, diag, diagonal, kron, reshape, einsum, sqrt, exp, conj, real
from torch import tile as repmat

from textremo_toolbox import *

eps = torch.finfo().eps

try:
    from ..vb.VB import VB
except:
    from .VB import VB

class VBNet(VB, nn.Module):
    
    def __init__(self, constel, modu, frame, pul, nTimeslot, nSubcarr, *args, B=None):
        nn.Module.__init__(self)
        VB.__init__(self, constel, modu, frame, pul, nTimeslot, nSubcarr, *args, B=B)
        
        constel = torch.as_tensor(constel).squeeze()
        self.register_buffer("constel_B_row", self.constel.repeat(self.B, 1, 1))
        
        
    '''
    init
    '''
    def init(self):
        self.register_buffer("constel_B", torch.zeros(3,2))
        
        