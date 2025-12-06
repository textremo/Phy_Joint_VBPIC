import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import FlexGRU


inDim = 4
hDim = 10


class MyModel(nn.Module):
    def __init__(self, inDim, hDim):
        super().__init__()
        self.gru = FlexGRU(inDim, hDim)
        self.ln = nn.Linear(hDim, inDim)
    def forward(self, x, h):
        h_new = self.gru(x, h)
        return self.ln(h_new)


model = MyModel(inDim, hDim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)


model.train()

for t in range(10):

    x = torch.randn(5, 3, inDim)
    h = torch.zeros(5, 3, hDim)
    
    x_pred = model(x, h)
    loss = criterion(x_pred, x)
    
    # back propagate
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
print("FlewGRU pass!")