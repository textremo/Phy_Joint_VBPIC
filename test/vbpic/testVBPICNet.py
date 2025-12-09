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
genconfig("OTFS", "EMBED", "toy-p1")

SNR_d = SNR_d[0];
No = 10**(-SNR_d/10);
N_fram = N_frams[0];
pil_pow = 10**((SNR_p - SNR_d)/10);
pil_thr = 3*sqrt(No);

## OTFS
constel = np.asarray(constel)
# generate data
xDD_syms_ids = randi(3,size=[B, data_len,1])
xDD_syms = constel[xDD_syms_ids]
# data to rg
rg = OTFSResGrid(M, N, batch_size=B)
rg.setPulse2Recta()
rg.setPilot2Center(1, 1)
rg.setGuard(gln_len, glp_len, gkn_len, gkp_len)
rg.map(xDD_syms, pilots_pow=pil_pow)
# pass the channel
otfs = OTFS(batch_size=B)
otfs.modulate(rg)
otfs.setChannel(p, lmax, kmax)
otfs.passChannel(No)
his, lis, kis = otfs.getCSI()
H_DD = otfs.getChannel()

'''
VBPICNet
'''
vbpicnn = VBPICNet(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, N, M, B=B)