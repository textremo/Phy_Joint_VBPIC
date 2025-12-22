import matplotlib.pyplot as plt
import gc
import scipy.io
import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy.random import randn, randint as randi
from numpy import real, imag
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from textremo_phy_mod_otfs import OTFS, OTFSResGrid
from config.genconfig import genconfig
from util import Utils
from src import *

torch.set_default_dtype(torch.float64)
dev = torch.device('cpu')
#dev = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')


B = 32
N_epo = 50000

genconfig("OTFS", "SP_REP_DELAY", "toy")
Es_d = 1
Es_p = 10**((SNR_p - SNR_d)/10)
No = 10**(-SNR_d/10)
N_fram = N_frams
constel = np.asarray(constel)
constel_r = np.asarray(constel_real)
dataLocs = np.ones([N, M])

'''
CPE->Tx
'''
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, N, M);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# CPE
cpe = CPE(otfsconfig, lmax, kmax, Es_d, No, B=B);
# pilots
Xp = cpe.genPilots(Es_p);
'''
OTFS frame
'''
#xDD_syms_ids = randi(3,size=[B, data_len,1])
#xDD_syms = constel[xDD_syms_ids]
xDD_syms_r_ids = randi(len(constel_r),size=[B, data_len,1], dtype=np.int64)
xDD_syms_i_ids = randi(len(constel_r),size=[B, data_len,1], dtype=np.int64)
xDD_syms = constel_r[xDD_syms_r_ids] + 1j*constel_r[xDD_syms_i_ids]
X_DD_syms = np.reshape(xDD_syms, [B, N, M])
X_DD = X_DD_syms + Xp
#X_DD = Xp
xDD = np.reshape(X_DD, [B, N*M, 1])
'''
OTFS Tx -> Rx
'''
# Tx
rg = OTFSResGrid(M, N, batch_size=B)
rg.setPulse2Recta()
rg.setContent(X_DD)

# pass the channel
otfs = OTFS(batch_size=B)
otfs.modulate(rg)
otfs.setChannel(p, lmax, kmax)
#otfs.passChannel(No)
otfs.passChannel(0)
his, lis, kis = otfs.getCSI()
H_DD = otfs.getChannel()
# Rx
rg_rx = otfs.demodulate()
Y_DD = rg_rx.getContent()
yDD = np.reshape(Y_DD, [B, N*M, 1])
'''
CPE->Rx
'''
h, hv, hm = cpe.estPaths(Y_DD, is_all=True)
his_full = Utils.realH2Hfull(kmax, lmax, his, lis, kis, batch_size=B);
# his_est, his_est_var, lis_est, kis_est = cpe.estPaths(Y_DD);
'''
to NN
'''
xDD_r_ids = torch.from_numpy(np.concat([xDD_syms_r_ids, xDD_syms_i_ids], -2).squeeze(-1)).to(dev)
H_DD_r = np.concat([np.concat([real(H_DD), -imag(H_DD)], -1), np.concat([imag(H_DD), real(H_DD)], -1)], -2)
H_DD_r = torch.from_numpy(H_DD_r).to(dev)

'''
VBPICNet
'''
vbpicnn = VBPICNet(Modu.MODU_OTFS_SP_REP_DELAY, Modu.FT_CP, Modu.PUL_RECTA, N, M, B=B, dev=dev)
vbpicnn.setCSI(1/p, p, kmax, lmax)
vbpicnn.setRef(Xp[0])
vbpicnn.setConstel(constel)
vbpicnn.setDataLoc(dataLocs)
vbpicnn.setNN()
'''
nn
'''
cc_ce = torch.nn.CrossEntropyLoss(reduction="mean").to(dev)
cc_ce2 = torch.nn.CrossEntropyLoss(reduction="sum").to(dev)
cc_mse = torch.nn.MSELoss(reduction='none').to(dev)
optimizer = torch.optim.Adam(vbpicnn.parameters(),lr=1e-4, weight_decay=1e-5)
l = 0.3

# train
vbpicnn.to(dev)
vbpicnn.train()
losses = zeros([N_epo, 1])
#plt.figure(figsize=(8, 6))
for t in range(N_epo):
    xDD_r_prob = vbpicnn(Y_DD, h, hv, hm, No, H=H_DD)
    loss_ce = 0
    l#oss_mse = cc_mse(H, H_DD_r).mean()
    for i in range(data_len):
        loss_ce = loss_ce + cc_ce(xDD_r_prob[:, i, :], xDD_r_ids[:, i])
    #loss = loss_mse * l + (1-l)*loss_ce
    loss = loss_ce
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    loss = loss.to('cpu').detach().numpy()
    losses[t] = loss
    
    #plt.plot(arange(N_epo), losses, 'b-', linewidth=2, label='loss')
    
    xDD_r_prob = xDD_r_prob.detach()
    
    xDD_est = vbpicnn.symmap(xDD_r_prob).numpy()
    ser = 1- np.sum((xDD_est == xDD_syms))/B/data_len
    
    
    
    
    print("%06d: loss=%e, SER=%e"%(t, loss, ser))

vbpicnn = None
del vbpicnn
gc.collect()
torch.cuda.empty_cache()

# Ts = vbpicnn.Ts.numpy()
# phi_rows = []
# for i in range(vbpicnn.pmax):
#     phi_rows.append(abs(Ts[i]) @ xDD)
# phi = np.concat(phi_rows, -1)
# phi2, _ = vbpicnn.x2P(xDD, xDD)


# yDD_diff_che = abs(yDD - phi @ his_full[..., None])
# yDD_diff_che_max = np.max(yDD_diff_che)


# H_DD_full = otfs.getChannel(his_full, repmat(vbpicnn.lis.numpy(), [B, 1]), repmat(vbpicnn.kis.numpy(), [B, 1]))

# yDD_diff_detect = abs(yDD - H_DD_full @ xDD)
# yDD_diff_detect_max = np.max(yDD_diff_detect)


# H_DD_full2 = zeros([B, N*M, N*M], dtype=complex)
# for i in range(vbpicnn.pmax):
#     hi = his_full[..., i]
#     H_DD_full2 = H_DD_full2 + hi.reshape(-1, 1, 1) * Ts[i];

# yDD_diff_detect2  = abs(yDD - H_DD_full2 @ xDD)
# yDD_diff_detect_max2 = np.max(yDD_diff_detect2)





