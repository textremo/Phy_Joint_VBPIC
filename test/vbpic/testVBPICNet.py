import gc
import scipy.io
import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy.random import randn, randint as randi
import torch
import torch.nn as nn
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


B = 5
genconfig("OTFS", "SP_REP_DELAY", "toy")
Es_d = 1
Es_p = 10**((SNR_p - SNR_d)/10)
No = 10**(-SNR_d/10)
N_fram = N_frams
constel = np.asarray(constel)
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
xDD_syms_ids = randi(3,size=[B, data_len,1])
xDD_syms = constel[xDD_syms_ids]
X_DD_syms = np.reshape(xDD_syms, [B, N, M])
X_DD = X_DD_syms + Xp
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
VBPICNet
'''
vbpicnn = VBPICNet(Modu.MODU_OTFS_SP_REP_DELAY, Modu.FT_CP, Modu.PUL_RECTA, N, M, B=B, dev=dev)
vbpicnn.setCSI(kmax, lmax)
vbpicnn.setRef(Xp[0])
vbpicnn.setConstel(constel)
vbpicnn.to(dev)
vbpicnn.setDataLoc(dataLocs)

vbpicnn.detect(Y_DD, h, hv, hm, No)

vbpicnn = None
del vbpicnn
gc.collect()
torch.cuda.empty_cache()

# Ts = vbpicnn.Ts.numpy()

# phi_rows = []
# for i in range(vbpicnn.pmax):
#     phi_rows.append(Ts[i] @ xDD)
# phi = np.concat(phi_rows, -1)
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





