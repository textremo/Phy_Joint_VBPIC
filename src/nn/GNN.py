import torch
from torch import as_tensor as arr
from torch import zeros, cat
import torch.nn as nn
try:
    from ..nn.FlexLinear import FlexLinear
    from ..nn.FlexGRU import FlexGRU
    from ..nn.TDGRU import TDGRU
except:
    from FlexLinear import FlexLinear
    from FlexGRU import FlexGRU
    from TDGRU import TDGRU
    
class GNN(nn.Module):
    FTYPE_BASE = 0;                     # basic features
    FTYPE_ALVA = 1;                     # ALVA version features on GRU
    FTYPES = [FTYPE_BASE, FTYPE_ALVA]
    
    NTYPE_CHE = 0
    NTYPE_DET = 1
    NTYPES = [NTYPE_CHE, NTYPE_DET]
    
    
    '''
    @iter_num:              the iteration number
    @node_num:              the number of node
    @feat_n_num_init:       node feature number(init)
    @feat_n_num:            node feature number(i.e., constel_len)
    @constel:               the constellation
    @nn_num:                the neuron number
    '''
    def __init__(self, iter_num, node_num, feat_n_num_init, constel, nn_num=64, *, ntype=NTYPE_CHE, ftype=FTYPE_ALVA):
        super().__init__()
        self.register_buffer("constel", constel)
        self.register_buffer("constel_len",  arr(constel.shape[-1]))
        feat_n_num = self.constel_len*2
        
        self.iter_num = iter_num
        self.u_is_ids = [i for i in range(node_num) for j in range(node_num) if i != j]
        self.u_js_ids = [j for i in range(node_num) for j in range(node_num) if i != j]
        self.mi_shape = [-1, node_num, node_num-1, feat_n_num]
        self.ntype = ntype
        self.ftype = ftype
        self.contel_len = feat_n_num
        
        # node initial
        self.fc1a = FlexLinear(feat_n_num_init, feat_n_num)
        # factor to variable
        self.fc2a = FlexLinear(feat_n_num*2+2, nn_num)
        self.fc2b = FlexLinear(nn_num, nn_num//2)
        self.fc2c = FlexLinear(nn_num//2, feat_n_num)
        self.a2 = nn.GELU()
        # variable to factor
        if ftype == self.FTYPE_BASE:
            gru_fin_num = feat_n_num
        if ftype == self.FTYPE_ALVA:
            gru_fin_num = feat_n_num + 2
        if ntype == self.NTYPE_CHE:
            gru_act = FlexGRU.ACT_GELU
        if ntype == self.NTYPE_DET:
            gru_act = FlexGRU.ACT_TANH
        self.fc3a = FlexGRU(gru_fin_num, nn_num, newAct=gru_act)
        #self.fc3a = FlexGRU(gru_fin_num, nn_num, newAct=FlexGRU.ACT_TANH)
        #self.fc3a = TDGRU(torch.nn.GRUCell, gru_fin_num, nn_num)
        
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
    @h:             the hidden states,       [B, node_num, nn_num]
    <opt>
    @node_mean:     the estimated mean,      [B, node_num, 1]
    @node_vari:     the estimated variance,  [B, node_num, 1]
    @last:          false
    '''
    def forward(self, node0, edge, h=None, *, node_mean=None, node_vari=None, last=False):
        # node init
        node = self.fc1a(node0)
        # GNN
        for t in range(self.iter_num):
            # factor to variable (node -> edge)
            u_ij = torch.cat([node[..., self.u_is_ids, :], node[..., self.u_js_ids, :], edge], -1) 
            # edge mlp
            m_ij = self.a2(self.fc2a(u_ij))
            m_ij = self.a2(self.fc2b(m_ij))
            m_ij = self.a2(self.fc2c(m_ij))
            # variable to factor (edge -> node)
            m_i = m_ij.contiguous().view(self.mi_shape).sum(-2)
            if self.ftype == self.FTYPE_ALVA:
                m_i = cat([m_i, node_mean, 1/node_vari], -1)
            h = self.fc3a(m_i, h)
            node = self.fc3b(h)
        # readout
        R = self.fc4c(self.fc4b(self.fc4a(node)))
        # feature aggregation - CHE
        if self.ntype == self.NTYPE_CHE:
            R = self.fc5b(self.fc5a(R))
            return R[..., 0][..., None], 1/R[..., 1][..., None]
        # feature aggregation - DET
        if self.ntype == self.NTYPE_DET:
            x = R[..., :self.constel_len]
            prec = R[..., self.constel_len:]
            logits = -0.5*prec*torch.square(x - self.constel)
            P = self.a6(logits)
            if last:
                return logits, P
            x_total = torch.sum(P*x, -1, keepdim=True)
            #x_total = torch.sum(P*self.constel, -1, keepdim=True)
            v_total = torch.sum(P*(1/prec + torch.square(x - x_total)), -1, keepdim=True)
            return x_total, v_total
        # if softmax:
        #     R = self.a6(R)