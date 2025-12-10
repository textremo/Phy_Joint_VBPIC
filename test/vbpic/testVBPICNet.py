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

# his_est, his_est_var, lis_est, kis_est = cpe.estPaths(Y_DD);




'''
VBPICNet
'''
vbpicnn = VBPICNet(Modu.MODU_OTFS_SP_REP_DELAY, Modu.FT_CP, Modu.PUL_RECTA, N, M, B=B)
vbpicnn.setConstel(constel)
vbpicnn.setDataLoc(dataLocs);
vbpicnn.setCSI(kmax, lmax)
vbpicnn.setRef(Xp[0]);

vbpicnn.detect(Y_DD, h, hv, hm, No)

