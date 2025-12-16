import torch
from torch import arange, ones, zeros, eye, diag, diagonal, kron, reshape, where, einsum, sqrt, exp, conj
import torch.nn as nn
try:
    from ..nn.FlexLinear import FlexLinear
    from ..nn.FlexGRU import FlexGRU
except:
    from FlexLinear import FlexLinear
    from FlexGRU import FlexGRU
    
class GNN(nn.Module):
    '''
    @iter_num:              the iteration number
    @node_num:              the number of node
    @feat_n_num_init:       node feature number(init)
    @feat_n_num:            node feature number(i.e., 2*constel_len, [mean, 1/vari])
    @nn_num:                the neuron number
    '''
    def __init__(self, iter_num, node_num, feat_n_num_init, feat_n_num, nn_num=64):
        super().__init__()
        self.iter_num = iter_num
        self.u_is_ids = [i for i in range(node_num) for j in range(node_num) if i != j]
        self.u_js_ids = [j for i in range(node_num) for j in range(node_num) if i != j]
        self.mi_shape = [-1, node_num, node_num-1, feat_n_num]
        
        # node initial
        self.fc1a = FlexLinear(feat_n_num_init, feat_n_num)
        
        # factor to variable
        self.fc2a = FlexLinear(feat_n_num*2+2, nn_num)
        self.fc2b = FlexLinear(nn_num, nn_num//2)
        self.fc2c = FlexLinear(nn_num//2, feat_n_num)
        self.a2 = nn.GELU()
        
        # variable to factor
        self.fc3a = FlexGRU(feat_n_num, nn_num, newAct=FlexGRU.ACT_GELU)
        self.fc3b = FlexLinear(nn_num, feat_n_num)
        
        # readout
        self.fc4a = FlexLinear(feat_n_num, nn_num)
        self.fc4b = FlexLinear(nn_num, nn_num//2)
        self.fc4c = FlexLinear(nn_num//2, feat_n_num)
        
        # feature aggregation
        self.fc5a = FlexLinear(feat_n_num, feat_n_num//2)
        self.fc5b = FlexLinear(feat_n_num//2, 2)
        
        # softmax
        self.a6 = nn.Softmax(dim=-1)
        
    '''
    @node0:         the node features(init), [B, node_num, feat_n_num_init]
    @edge:          the edge features,       [B, node_num*(node_num-1), 2]
    @node_mean:     the estimated mean,      [B, node_num, 1]
    @node_vari:     the estimated variance,  [B, node_num, 1]
    @h:             the hidden states,       [B, node_num, nn_num]
    <opt>
    @aggr:          feature aggregation (defalut: False)
    @softmax:       softmax (default: False)
    '''
    def forward(self, node0, edge, node_mean, node_vari, h, *, aggr=False, softmax=False):
        # node init
        node = self.fc1a(node0)
        # GNN
        for t in range(self.iter_num):
            # factor to variable (node -> edge)
            u_ij = torch.cat([node[..., self.u_is_ids, :], node[..., self.u_js_ids, :], edge], -1) 
            self.n2e(node, edge)
            # edge mlp
            m_ij = self.a2(self.fc2a(u_ij))
            m_ij = self.a2(self.fc2b(m_ij))
            m_ij = self.a2(self.fc2c(m_ij))
            # variable to factor (edge -> node)
            m_i = m_ij.contiguous().view(self.mi_shape).sum(-2)
            h = self.fc3a(m_i, h)
            node = self.fc3b(h)
        # readout
        R = self.fc4c(self.fc4b(self.fc4a(node)))
        # feature aggregation
        if aggr:
            R = self.fc5b(self.fc5a(R))
        if softmax:
            R = self.a6(R)
        return R