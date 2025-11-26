import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy.random import randn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import *;
from textremo_phy_mod_otfs import OTFS, OTFSResGrid



print("case 0x (common)");
'''
case 01 - init
'''
print("     01 (init)");
nTimeslot = 4;
nSubcarr = 4;
dataLocs = [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]];
refSig = [[0, 0, 0, 0], [0, -1+1j, -1-1j, 0], [0, 1+1j, 1-1j, 0], [0, 0, 0, 0]];
lmax = 2;
kmax = 1;
csiLim = [lmax, kmax];
lis = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)).astype(int);
kis = np.arange(-kmax, kmax+1).repeat(lmax+1).astype(int);
h = [0, 0, 1, 0, 2, 0, 0, 0, 0];

try:
    modu = Modu(Modu.MODU_OTFS_FULL, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr);
    modu.setDataLoc(dataLocs);
    modu.setRef(refSig);
except:
    print("        - illegal refSig pass.");
modu = Modu(Modu.MODU_OTFS_FULL, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr);
modu.setDataLoc(dataLocs);
modu = Modu(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, csiLim);
modu.setDataLoc(dataLocs);
modu.setRef(refSig);
print("        - all pass.");


print("case 2x (OTFS)");
'''
case 20 - h2H
'''
print("     21 (h2H)", end="")
B = 2
nTimeslot = 6;
nSubcarr = 6;
dataLocs = ones([nTimeslot, nSubcarr]); dataLocs[2:4, 2:4] = 0;
refSig = zeros([nTimeslot, nSubcarr], dtype=complex); 
refSig[2:4, 2:4] = [[-1-1j, -1+1j], [1-1j, 1+1j]];
lmax = 2;
kmax = 1;
csiLim = [lmax, kmax];
lis = kron(arange(lmax+1), ones(2*kmax+1)).astype(int);
kis = arange(-kmax,kmax+1).repeat(lmax+1).astype(int);

h = randn(B,9) + randn(B,9)*1j;

modu = Modu(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, csiLim, B=B);
modu.setDataLoc(dataLocs);
modu.setRef(refSig);
[H, Hv] = modu.h2H(h);

rg = OTFSResGrid(nSubcarr, nTimeslot, batch_size=B);
rg.setPulse2Recta();
rg.setContent(repmat(refSig, [B,1,1]));
otfs = OTFS(batch_size=B);
otfs.modulate(rg);
otfs.setChannel(h, repmat(lis, [B, 1]), repmat(kis, [B, 1]));
Hdd = otfs.getChannel();
if np.max(abs(H - Hdd)) <= 1e-14:
    print(": Recta pass at %e."%np.max(abs(H - Hdd)));
else:
    print(": Recta fail at %e."%np.max(abs(H - Hdd)));
    
'''
case 21 - refSig
'''
print("     21 (refSig)", end="");
otfs.passChannel(0);
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
Y_DD_cherng = Y_DD[..., modu.pilCheRng[0]:modu.pilCheRng[1]+1, modu.pilCheRng[2]:modu.pilCheRng[3]+1];
yDD_cherng = reshape(Y_DD_cherng, [B, -1], order='F');

Phi_p = modu.ref2Phi();
yDD_cherng_est = (Phi_p @ h[..., None]).squeeze(-1);
if np.max(abs(yDD_cherng - yDD_cherng_est)) < 1e-14:
    print(": pass at %e."%np.max(abs(yDD_cherng - yDD_cherng_est)));