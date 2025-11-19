clear;
clc;
%% Config
genconfig("OTFS", "EMBED", "toy-p4");
SNR_d = SNR_d(1);
No = 10.^(-SNR_d/10);
N_fram = N_frams(1);
pil_pow = 10^((SNR_p - SNR_d)/10);
pil_thr = 3*sqrt(No);


%% OTFS
% generate data
nbits = randi([0,1],data_len*M_bits,1);
xDD_syms = qammod(nbits, M_mod,'InputType','bit','UnitAveragePower',true);
% data to rg
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setPilot2Center(pl_len, pk_len);
rg.setGuard(gln_len, glp_len, gkn_len, gkp_len);
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
% to full his
his = Utils.realH2Hfull(kmax, lmax, his, lis, kis);

%% VB - iter 1
dataLocs = rg.getContentDataLocsMat();
refSig = zeros(N, M); refSig(4:5,4:5) = (1+1j)*sqrt(pil_pow/2);
csiLim = [lmax, kmax];
Y_DD = rg_rx.getContent();
vb = VB(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, N, M, csiLim);
vb.setDataLoc(dataLocs);
vb.setRef(refSig);

his_est1 = vb.che(Y_DD, "No", No);
his_est1 = his_est1.';
his_est2 = vb.che(Y_DD);
his_est2 = his_est2.';
hm = abs(his) > 0;
% print
disp("iter 1");
est1_diff = abs(his_est1(hm) - his(hm));
est1_err = abs(his_est1(~hm));
est2_diff = abs(his_est2(hm) - his(hm));
est2_err = abs(his_est2(~hm));
disp(" - vb(know No)");
fprintf("    diff: %e\n", max(est1_diff));
fprintf("    err: %e\n", max(est1_err));
disp(" - vb");
fprintf("    diff: %e\n", max(est1_diff));
fprintf("    err: %e\n", max(est1_err));
%% VB - iter N
user_input = input('iter-N: please press Enter to continue: ', 's');
if ~isempty(user_input)
    return;
end

clear;
genconfig("OTFS", "EMBED", "toy-p1");
pmax = (lmax+1)*(2*kmax+1);

vb_diff_max = zeros(length(SNR_d), 1);
vb_err_max = zeros(length(SNR_d), 1);
vb_diff_aver = zeros(length(SNR_d), 1);
vb_err_aver = zeros(length(SNR_d), 1);

vb2_diff_max = zeros(length(SNR_d), 1);
vb2_err_max = zeros(length(SNR_d), 1);
vb2_diff_aver = zeros(length(SNR_d), 1);
vb2_err_aver = zeros(length(SNR_d), 1);
for i = 1:length(SNR_d)
    SNR_di = SNR_d(i);
    if i == length(SNR_d)
        fprintf("  - SNR: %d\n", SNR_di);
    else
        fprintf("  - SNR: %d, ", SNR_di);
    end
    
    No = 10.^(-SNR_di/10);
    N_fram = N_frams(i);
    pil_pow = 10^((SNR_p - SNR_di)/10);
    pil_thr = 3*sqrt(No);
    % store tmp variables
    vb_diff_max_tmp = zeros(N_fram, 1);
    vb_err_max_tmp = zeros(N_fram, 1);
    vb_diff_aver_tmp = zeros(N_fram, 1);
    vb_err_aver_tmp = zeros(N_fram, 1);
    vb2_diff_max_tmp = zeros(N_fram, 1);
    vb2_err_max_tmp = zeros(N_fram, 1);
    vb2_diff_aver_tmp = zeros(N_fram, 1);
    vb2_err_aver_tmp = zeros(N_fram, 1);
    % frames
    parfor N_frami = 1:N_fram
        % generate data
        nbits = randi([0,1],data_len*M_bits,1);
        xDD_syms = qammod(nbits, M_mod,'InputType','bit','UnitAveragePower',true);
        % data to rg
        rg = OTFSResGrid(M, N);
        rg.setPulse2Recta();
        rg.setPilot2Center(1, 1);
        rg.setGuard(gln_len, glp_len, gkn_len, gkp_len);
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
        % to full his
        his = Utils.realH2Hfull(kmax, lmax, his, lis, kis);
        hm = abs(his) > 0;

        % VB
        dataLocs = rg.getContentDataLocsMat();
        refSig = zeros(N, M); refSig(4:5,4:5) = (1+1j)*sqrt(pil_pow/2);
        csiLim = [lmax, kmax];
        Y_DD = rg_rx.getContent();
        vb = VB(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, N, M, csiLim);
        vb.setDataLoc(dataLocs);
        vb.setRef(refSig);
        his_est1 = vb.che(Y_DD, "No", No);
        his_est1 = his_est1.';
        his_est2 = vb.che(Y_DD);
        his_est2 = his_est2.';

        vb_diff_max_tmp(N_frami) = max(abs(his_est1(hm) - his(hm)));
        vb_err_max_tmp(N_frami) = max(abs(his_est1(~hm)));
        vb_diff_aver_tmp(N_frami) = mean(abs(his_est1(hm) - his(hm)));
        vb_err_aver_tmp(N_frami) = mean(abs(his_est1(~hm)));
        
        vb2_diff_max_tmp(N_frami) = max(abs(his_est2(hm) - his(hm)));
        vb2_err_max_tmp(N_frami) = max(abs(his_est2(~hm)));
        vb2_diff_aver_tmp(N_frami) = mean(abs(his_est2(hm) - his(hm)));
        vb2_err_aver_tmp(N_frami) = mean(abs(his_est2(~hm)));
    end

    vb_diff_max(i) = max(vb_diff_max_tmp);
    vb_err_max(i) = max(vb_err_max_tmp);
    vb_diff_aver(i) = mean(vb_diff_aver_tmp);
    vb_err_aver(i) = mean(vb_err_aver_tmp);

    vb2_diff_max(i) = max(vb2_diff_max_tmp);
    vb2_err_max(i) = max(vb2_err_max_tmp);
    vb2_diff_aver(i) = mean(vb2_diff_aver_tmp);
    vb2_err_aver(i) = mean(vb2_err_aver_tmp);
end

figure;
subplot(2, 2, 1);
plot(SNR_d, 10*log10(vb_diff_max)); hold on;
plot(SNR_d, 10*log10(vb2_diff_max)); 
hold off;
grid on; xlabel("SNR(dB)"); title("Diff(max)"); legend("know No", "update \alpha");
subplot(2, 2, 2);
plot(SNR_d, 10*log10(vb_err_max)); hold on;
plot(SNR_d, 10*log10(vb2_err_max));
hold off;
grid on;xlabel("SNR(dB)"); title("Err(max)"); legend("know No", "update \alpha");
subplot(2, 2, 3);
plot(SNR_d, 10*log10(vb_diff_aver)); hold on;
plot(SNR_d, 10*log10(vb2_diff_aver));
hold off;
grid on;xlabel("SNR(dB)"); title("Diff(aver)"); legend("know No", "update \alpha");
subplot(2, 2, 4);
plot(SNR_d, 10*log10(vb_err_aver)); hold on;
plot(SNR_d, 10*log10(vb2_err_aver));
hold off;
grid on;xlabel("SNR(dB)"); title("Err(aver)"); legend("know No", "update \alpha");