import torch
import torch.nn as nn


class FlexLinear(nn.Module):    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.f = nn.Linear(in_features, out_features)
        self.out_features = out_features
    
    def __call__(self, x):
        x_shape = list(x.shape)
        y_shape = x_shape.copy()
        y_shape[-1] = self.out_features
        ndim = x.ndim
        if ndim > 2:
            x = x.contiguous().view(-1, x_shape[-1])     
        y = self.f(x)
        if ndim > 2:
            y = y.contiguous().view(y_shape)
        return y
        