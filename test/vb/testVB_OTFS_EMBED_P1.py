import scipy.io
import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy.random import randn, randint as randi
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from textremo_phy_mod_otfs import OTFS, OTFSResGrid
from config.genconfig import genconfig
from util import Utils
from src import *

B = 2

user_input = input('Do you use matlab data as input? please press Enter as yes: ')
if user_input:
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
    # Rx
    rg_rx = otfs.demodulate()
    yDD, his_est0, lis_est0, kis_est0 = rg_rx.demap(threshold=pil_thr);
    # to full his
    his = Utils.realH2Hfull(kmax, lmax, his, lis, kis, batch_size=B);
    his_est0 = Utils.realH2Hfull(kmax, lmax, his_est0, lis_est0, kis_est0, batch_size=B);

    ## VB - iter 1
    dataLocs = rg.getContentDataLocsMat();
    refSig = zeros([N, M]).astype(complex); refSig[3,3] = (1+1j)*sqrt(pil_pow/2);
    Y_DD = rg_rx.getContent();
else:
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

vb = VB(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, N, M, B=B);
vb.setConstel(constel)
vb.setDataLoc(dataLocs);
vb.setCSI(kmax, lmax)
vb.setRef(refSig);

his_est1 = vb.che(Y_DD, No=No);
his_est2 = vb.che(Y_DD);
hm = abs(his) > 0;

# print
if user_input:
    print("iter 1");
    est0_diff = abs(his_est0[hm] - his[hm]);
    est1_diff = abs(his_est1[hm] - his[hm]);
    est1_err = abs(his_est1[~hm]);
    est2_diff = abs(his_est2[hm] - his[hm]);
    est2_err = abs(his_est2[~hm]);
    print(" - threshold diff: %e"%max(est0_diff));
    print(" - vb(know No)");
    print("    diff: %e"%max(est1_diff));
    print("    err: %e"%max(est1_err));
    print(" - vb");
    print("    diff: %e"%max(est1_diff));
    print("    err: %e"%max(est1_err));
else:
    his_est1_diff = abs(his_est1_mat - his_est1)
    his_est2_diff = abs(his_est2_mat - his_est2)
    
    print("his_est1 diff mat %e"%np.max(his_est1_diff))
    print("his_est2 diff mat %e"%np.max(his_est2_diff))