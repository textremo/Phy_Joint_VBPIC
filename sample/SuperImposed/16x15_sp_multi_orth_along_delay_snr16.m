clear
clc

% configuration
K = 15;     % timeslote number
L = 16;     % subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = 14;
SNR_p = 37;
No = 10^(-SNR_d/10);
Es_d = 1;
Es_p = 10^((SNR_p - SNR_d)/10);
% channel configuration
p = 6;
lmax = 3;
kmax = 5;
% JPIC config
iter_num = 20;

% init generalised variables
% config
oc = OTFSConfig();
oc.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, K, L);
oc.setPul(OTFSConfig.PUL_TYPE_RECTA);
% OTFS module
otfs = OTFS();
% CPE
cpe = CPE(oc, lmax, kmax, Es_d, No);
% pilots
X_p = cpe.genPilots(Es_p);

% Tx
% generate symbols
sym_idx = randi(4, K*L, 1);
syms_vec = constel(sym_idx);
syms_mat = reshape(syms_vec, L, K).';
% generate X_DD
X_DD = syms_mat + X_p;
xDD = reshape(X_DD.', K*L, 1);
% generate
rg = OTFSResGrid(L, K);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

% channel
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(No);
[his, lis, kis] = otfs.getCSI();
Hdd = otfs.getChannel();

% Rx
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = reshape(Y_DD.', K*L, 1);

% initial CHE
[his_est, his_var, his_est_mask] = cpe.estPaths(Y_DD, is_all=true);
[kmin, kmax] = cpe.getKRange();

% joint detection
jpic = JPIC(oc, constel, 1/p, lmax, kmin, kmax, "iter_num", iter_num);
jpic.setSdBsoMealCalInit2MMSE();
[x_est, Hdd_est] = jpic.detect(Y_DD, X_p, his_est, his_var, his_est_mask, No, "sym_map", true);

% compare
ser = sum(xDD ~= x_est)/(K*L);
ncme = sum(abs(Hdd_est - Hdd), "all")/(K*L)^2;
fprintf("SNR %d (Pilot SNR %d)\n", SNR_d, SNR_p);
fprintf("  - SER: %.8f\n", ser);
fprintf("  - NCME: %.8f\n", ncme);
