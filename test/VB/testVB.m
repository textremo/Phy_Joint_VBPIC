clear;
clc;
%% Config
genconfig("OTFS", "EMBED", "toy-p1");

configOTFSEmbedP1;
SNR_d = SNR_ds(1);
No = Nos(1);
N_fram = N_frams(1);
pil_pow = 10^((SNR_p - SNR_d)/10);
pil_thr = 3*sqrt(No);
fprintf("SNR=%d\n",SNR_d);

%% OTFS
% generate data
nbits = randi([0,1],sig_len*M_bits,1);
xDD_syms = qammod(nbits, M_mod,'InputType','bit','UnitAveragePower',true);
% data to rg
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setPilot2Center(1, 1);
rg.setGuard(gdn_len, gdp_len, gkn_len, gkp_len);
rg.map(xDD_syms, "pilots_pow", pil_pow);
% pass the channel
otfs = OTFS();
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(No);
[his, lis, kis] = otfs.getCSI();
H_DD = otfs.getChannel();
% Rx
rg_rx = otfs.demodulate();
[yDD, his_est0, lis_est0, kis_est0] = rg_rx.demap("threshold", pil_thr);


%% VB

dataLocs = rg.getContentDataLocsMat();
refSig = zeros(N, M); refSig(4,4) = (1+1j)*sqrt(pil_pow/2);
csiLim = [lmax, kmax];
Y_DD = rg_rx.getContent();
vb = VB(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, N, M, csiLim);
vb.setDataLoc(dataLocs);
vb.setRef(refSig);

his_est1 = vb.che(Y_DD, "No", No);
his_est1 = his_est1.';

%% 
his = Utils.realH2Hfull(kmax, lmax, his, lis, kis);
his_est0 = Utils.realH2Hfull(kmax, lmax, his_est0, lis_est0, kis_est0);

hm = abs(his) > 0;

est0_diff = abs(his_est0(hm) - his(hm));
est1_diff = abs(his_est1(hm) - his(hm));
est1_err = abs(his_est1(~hm));

fprintf(" - threshold diff: %e\n", max(est0_diff));
disp(" - vb");
fprintf("    diff: %e\n", max(est1_diff));
fprintf("    err: %e\n", max(est1_err));

