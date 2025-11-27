import scipy.io
import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy.random import randn, randint as randi
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from textremo_phy_mod_otfs import OTFS, OTFSResGrid
from config.genconfig import genconfig
from util import Utils
from src import *


B = 2

if 'mat_data' not in locals():
    mat_data = scipy.io.loadmat('../../_dist/test/testVB_OTFS_EMBED_P1.mat', 
                                variable_names=['N', 'M', 'No', 'lmax', 'kmax', 'dataLocs', 'refSig',
                                                'Y_DD', 'constel', 'his', 'his_est0', 'his_est1', 'his_est2'])
N = mat_data['N'].squeeze()
M = mat_data['M'].squeeze()
No = mat_data['No'].squeeze()
csiLim = [mat_data['lmax'].squeeze().astype(int), mat_data['kmax'].squeeze().astype(int)]
kmax = csiLim[1]
lmax = csiLim[0]
dataLocs = mat_data['dataLocs']
refSig = mat_data['refSig']
Y_DD = repmat(mat_data['Y_DD'], [B, 1, 1])
constel = mat_data['constel'].squeeze()
his = repmat(mat_data['his'], [B, 1])
his_est0 = repmat(mat_data['his_est0'], [B, 1])
his_est1_mat = repmat(mat_data['his_est1'], [B, 1, 1])
his_est2_mat = repmat(mat_data['his_est2'], [B, 1, 1])

'''
VBNet
'''
torch.set_default_dtype(torch.float64)
dev = torch.device('cpu')
#dev = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')

vbn = VBNet(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, N, M, B=B, dev=dev)
vbn.setConstel(constel)
vbn.setDataLoc(dataLocs);
vbn.setCSI(kmax, lmax)
vbn.setRef(refSig);
vbn.to(dev)

his_est1 = vbn.che(Y_DD, No=No).detach().to('cpu').numpy()
his_est2 = vbn.che(Y_DD).to('cpu').numpy()

hm = abs(his) > 0;
# print
# print("iter 1");
# est0_diff = abs(his_est0[hm] - his[hm]);
# est1_diff = abs(his_est1[hm] - his[hm]);
# est1_err = abs(his_est1[~hm]);
# est2_diff = abs(his_est2[hm] - his[hm]);
# est2_err = abs(his_est2[~hm]);
# print(" - threshold diff: %e"%max(est0_diff));
# print(" - vb(know No)");
# print("    diff: %e"%max(est1_diff));
# print("    err: %e"%max(est1_err));
# print(" - vb");
# print("    diff: %e"%max(est1_diff));
# print("    err: %e"%max(est1_err));

his_est1_diff = abs(his_est1_mat - his_est1)
his_est2_diff = abs(his_est2_mat - his_est2)

print("his_est1 diff mat %e"%np.max(his_est1_diff))
print("his_est2 diff mat %e"%np.max(his_est2_diff))