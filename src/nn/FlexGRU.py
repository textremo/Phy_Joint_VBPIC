import torch.nn as nn

class FlexGRU(nn.module):
    
    def __init__(self, inDim, hDim):
        super().__init__()
        