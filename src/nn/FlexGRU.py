import torch
import torch.nn as nn

class FlexGRU(nn.Module):
    ACT_RELU = 0;
    ACT_LEAKYRELU = 1;
    ACT_ELU = 2;
    ACT_GELU = 3;
    ACTs = [ACT_RELU, ACT_LEAKYRELU, ACT_ELU, ACT_GELU];
    
    '''
    constructor
    @inDim:         the input dimension
    @hDim:          the hidden state dimension
    @newActType:    the new gate activation function
    '''
    def __init__(self, inDim, hDim, *, newAct=ACT_GELU):
        super().__init__()
        
        # reset gate
        self.resetLNx = nn.Linear(inDim, hDim, bias=True)
        self.resetLNh = nn.Linear(hDim, hDim, bias=True)
        self.resetAct = torch.sigmoid
        # update gate
        self.updateLNx = nn.Linear(inDim, hDim, bias=True)
        self.updateLNh = nn.Linear(hDim, hDim, bias=True)
        self.updateAct = torch.sigmoid
        # new gate
        self.newLNx = nn.Linear(inDim, hDim, bias=True)
        self.newLNh = nn.Linear(hDim, hDim, bias=True)
        self.newAct = self.__act__(newAct)
        
    def __act__(self, actType):
        if actType not in self.ACTs:
            raise Exception("The activation function is not supported!!!")
        elif actType == self.ACT_RELU:
            return nn.ReLU()
        elif actType == self.ACT_LEAKYRELU:
            return nn.LeakyReLU()
        elif actType == self.ACT_ELU:
            return nn.ELU()
        elif actType == self.ACT_GELU:
            return nn.GELU()
        
    '''
    call
    @x: the data
    @h: the hidden state
    '''
    def __call__(self, x, h):
        x_shape = x.shape
        h_shape = h.shape
        
        if len(x_shape) > 2:
            x = x.contiguous().view(-1, x_shape[-1]) 
        if len(h_shape) > 2:
            h = h.contiguous().view(-1, h_shape[-1])
        r = self.resetAct(self.resetLNx(x) + self.resetLNh(h))
        z = self.updateAct(self.updateLNx(x) + self.updateLNh(h))
        n = self.newAct(self.newLNx(x) + r*self.newLNh(h))
        h_new = (1-z)*n + z*h
        if len(h_shape) > 2:
            h_new = h_new.contiguous().view(h_shape)
        return h_new
        