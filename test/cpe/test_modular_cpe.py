from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')

import numpy as np
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
from OTFSConfig import OTFSConfig
from CPE import CPE
from JPICNet import JPICNet
from Utils.utils import realH2Hfull

print("------------------------------------------------------------------------")
print("CPE\n")

'''
unbatched
'''
# configuration
K = 15;     # timeslote number
L = 16;     # subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = 14;
SNR_p = 37;
No = 10**(-SNR_d/10);
Es_d = 1;
Es_p = 10**((SNR_p - SNR_d)/10);
# channel configuration
p = 6;
lmax = 3;
kmax = 5;

# init generalised variables
# config
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, K, L);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS();
# CPE
cpe = CPE(otfsconfig, lmax, kmax, Es_d, No);
# pilots
X_p = cpe.genPilots(Es_p);

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(K*L));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [K, L]);
# generate X_DD
X_DD = X_p;
# generate
rg = OTFSResGrid(L, K);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();
his_full = realH2Hfull(kmax, lmax, his, lis, kis);

# Rx
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [K*L]);

# estimate the paths
# estimate the paths - positve 
his_est, his_est_var, lis_est, kis_est = cpe.estPaths(Y_DD);
his_full_est = realH2Hfull(kmax, lmax, his_est, lis_est, kis_est);
his_full_diff = abs(his_full_est - his_full);
# estimate the paths - all
his_full_est1, his_full_est1_var, his_mask =  cpe.estPaths(Y_DD, is_all=True);
lis_full_est1 = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)).astype(int);  # the delays on all possible paths
kis_full_est1 = np.tile(np.arange(-kmax, kmax+1), lmax+1);     
his_full_diff1 = abs(his_full_est1 - his_full_est);

print("- unbatched: CPE threshold (power): %f"%cpe.thres)
print("  - CHE result compare");
print("    - CHE check (less):")
diff_num_less = 0;
for li in range(lmax + 1):
    for ki in range(-kmax, kmax+1):
        his_shift = li*(2*kmax+1) + kmax + ki;
        y_pow = abs(his_full_est[his_shift]*cpe.pil_val)**2;
        if his_full_diff[his_shift] > 1e-13 and y_pow <= cpe.thres:
            diff_num_less += 1
            print(f"      - [{his_shift:2d}], origin: {his_full[his_shift]:+.4f}, est: {his_full_est[his_shift]:+.4f}")
print("    - CHE check (greater):")
diff_num_grea = 0;
for li in range(lmax + 1):
    for ki in range(-kmax, kmax+1):
        his_shift = li*(2*kmax+1) + kmax + ki;
        y_pow = abs(his_full_est[his_shift]*cpe.pil_val)**2;
        if his_full_diff[his_shift] > 1e-13 and y_pow > cpe.thres:
            diff_num_grea += 1
            print(f"      - [{his_shift:2d}], origin: {his_full[his_shift]:+.4f}, est: {his_full_est[his_shift]:+.4f}, diff: {his_full_diff[his_shift]: .4f}")
            

if diff_num_less + diff_num_grea != np.sum(his_full_diff > 1e-13):
    raise Exception("Difference not match!");
else:
    print(f"  - find {diff_num_less + diff_num_grea} difference, at max {np.max(his_full_diff):.4f}.")
print("  - CHE full");
if np.any( abs(his_full_est1[his_mask]) < 1e-13 ):
    raise Exception("Mask one not match!");
else:
    print("    - mask selected values are positive!");
if np.any( abs(his_full_est1[~his_mask])!= 0 ):
    raise Exception("Mask zero not match!");
else:
    print("    - mask unselected values are zeros!");
if np.max(his_full_diff1) > 0:
    raise Exception("his not match!");
else:
    print("    - his are the same!");

'''
batched - 1
'''
# configuration
K = 15;     # timeslote number
L = 16;     # subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = 14;
SNR_p = 37;
No = 10**(-SNR_d/10);
Es_d = 1;
Es_p = 10**((SNR_p - SNR_d)/10);
# batch
batch_size = 1;
# channel configuration
p = 6;
lmax = 3;
kmax = 5;

# init generalised variables
# config
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, K, L);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);
# CPE
cpe = CPE(otfsconfig, lmax, kmax, Es_d, No, B=batch_size);
# pilots
X_p = cpe.genPilots(Es_p);

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, K*L));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, K, L]);
# generate X_DD
X_DD = X_p;
# generate
rg = OTFSResGrid(L, K, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();
his_full = realH2Hfull(kmax, lmax, his, lis, kis, batch_size=batch_size);

# Rx
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [batch_size, K*L]);

# estimate the paths
# estimate the paths - positve 
his_est, his_est_var, lis_est, kis_est = cpe.estPaths(Y_DD);
his_full_est = realH2Hfull(kmax, lmax, his_est, lis_est, kis_est, batch_size=batch_size);
his_full_diff = abs(his_full_est - his_full);
# estimate the paths - all
his_full_est1, his_full_est1_var, his_mask =  cpe.estPaths(Y_DD, is_all=True);
lis_full_est1 = np.tile(np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)), [batch_size, 1]).astype(int);  # the delays on all possible paths
kis_full_est1 = np.tile(np.arange(-kmax, kmax+1), [batch_size, lmax+1]);     
his_full_diff1 = abs(his_full_est1 - his_full_est);



print("\n- batch - %d: CPE threshold (power): %f"%(batch_size, cpe.thres))
print("  - CHE result compare");
print("    - CHE check (less):")
diff_num_less = 0;
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if his_full_diff[bid, his_shift] > 1e-13 and y_pow <= cpe.thres:
                diff_num_less += 1
                print(f"      - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}")
print("    - CHE check (greater):")
diff_num_grea = 0;
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if his_full_diff[bid, his_shift] > 1e-13 and y_pow > cpe.thres:
                diff_num_grea += 1
                print(f"      - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}, diff: {his_full_diff[bid, his_shift]: .4f}")
                

if diff_num_less + diff_num_grea != np.sum(his_full_diff > 1e-13):
    raise Exception("Difference not match!");
else:
    print(f"  - find {diff_num_less + diff_num_grea} difference, at max {np.max(his_full_diff):.4f}.")
print("  - CHE full");
if np.any( abs(his_full_est1[his_mask])<1e-13 ):
    raise Exception("Mask one not match!");
else:
    print("    - mask selected values are positive!");
if np.any( abs(his_full_est1[~his_mask])!=0 ):
    raise Exception("Mask zero not match!");
else:
    print("    - mask unselected values are zeros!");
if np.max(his_full_diff1) > 0:
    raise Exception("his not match!");
else:
    print("    - his are the same!");

'''
batched - n
'''
# configuration
K = 15;     # timeslote number
L = 16;     # subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = 14;
SNR_p = 37;
No = 10**(-SNR_d/10);
Es_d = 1;
Es_p = 10**((SNR_p - SNR_d)/10);
# batch
batch_size = 10;
# channel configuration
p = 6;
lmax = 3;
kmax = 5;

# init generalised variables
# config
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, K, L);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);
# CPE
cpe = CPE(otfsconfig, lmax, kmax, Es_d, No, B=batch_size);
# pilots
X_p = cpe.genPilots(Es_p);

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, K*L));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, K, L]);
# generate X_DD
X_DD = X_p;
# generate
rg = OTFSResGrid(L, K, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();
his_full = realH2Hfull(kmax, lmax, his, lis, kis, batch_size=batch_size);

# Rx
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [batch_size, K*L]);

# estimate the paths
# estimate the paths - positve 
his_est, his_est_var, lis_est, kis_est = cpe.estPaths(Y_DD);
his_full_est = realH2Hfull(kmax, lmax, his_est, lis_est, kis_est, batch_size=batch_size);
his_full_diff = abs(his_full_est - his_full);
# estimate the paths - all
his_full_est1, his_full_est1_var, his_mask =  cpe.estPaths(Y_DD, is_all=True);
lis_full_est1 = np.tile(np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)), [batch_size, 1]).astype(int);  # the delays on all possible paths
kis_full_est1 = np.tile(np.arange(-kmax, kmax+1), [batch_size, lmax+1]);     
his_full_diff1 = abs(his_full_est1 - his_full_est);



print("\n- batch - n: CPE threshold (power): %f"%cpe.thres)
print("  - CHE result compare");
print("    - CHE check (less):")
diff_num_less = 0;
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if his_full_diff[bid, his_shift] > 1e-13 and y_pow <= cpe.thres:
                diff_num_less += 1
                print(f"      - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}")
print("    - CHE check (greater):")
diff_num_grea = 0;
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if his_full_diff[bid, his_shift] > 1e-13 and y_pow > cpe.thres:
                diff_num_grea += 1
                print(f"      - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}, diff: {his_full_diff[bid, his_shift]: .4f}")
                

if diff_num_less + diff_num_grea != np.sum(his_full_diff > 1e-13):
    raise Exception("Difference not match!");
else:
    print(f"  - find {diff_num_less + diff_num_grea} difference, at max {np.max(his_full_diff):.4f}.")
print("  - CHE full");
if np.any( abs(his_full_est1[his_mask])<1e-13 ):
    raise Exception("Mask one not match!");
else:
    print("    - mask selected values are positive!");
if np.any( abs(his_full_est1[~his_mask])!= 0 ):
    raise Exception("Mask zero not match!");
else:
    print("    - mask unselected values are zeros!");
if np.max(his_full_diff1) > 0:
    raise Exception("his not match!");
else:
    print("    - his are the same!");

print("------------------------------------------------------------------------")