clear
clc

fprintf("------------------------------------------------------------------------\n")
fprintf("CPE\n\n")

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

% init generalised variables
% config
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, K, L);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
% OTFS module
otfs = OTFS();
% CPE
cpe = CPE(otfsconfig, lmax, kmax, Es_d, No);
% pilots
X_p = cpe.genPilots(Es_p);

% Tx
% generate symbols
sym_idx = randi(4, K*L, 1);
syms_vec = constel(sym_idx);
syms_mat = reshape(syms_vec, L, K).';
% generate X_DD
X_DD = X_p;
% generate
rg = OTFSResGrid(L, K);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

% channel
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
[his, lis, kis] = otfs.getCSI();
H_DD = otfs.getChannel();
his_full = Utils.realH2Hfull(kmax, lmax, his, lis, kis);

% Rx
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = reshape(Y_DD.', K*L, 1);

% estimate the paths
% estimate the paths - positve 
[his_est, his_est_var, lis_est, kis_est] = cpe.estPaths(Y_DD);
his_full_est = Utils.realH2Hfull(kmax, lmax, his_est, lis_est, kis_est);
his_full_diff = abs(his_full_est - his_full);
% estimate the paths - all
[his_full_est1, his_full_est1_var, his_mask] =  cpe.estPaths(Y_DD, "is_all", true);
lis_full_est1 = kron(0:lmax, ones(1, 2*kmax + 1));          % the delays on all possible paths
kis_full_est1 = repmat(-kmax:kmax, 1, lmax+1);     
his_full_diff1 = abs(his_full_est1(:) - his_full_est(:));

fprintf("- CPE threshold (power): %f\n", cpe.thres)
fprintf("  - CHE result compare\n");
fprintf("    - CHE check (less):\n")
diff_num_less = 0;
for li = 0:lmax
    for ki = -kmax:kmax
        his_shift = li*(2*kmax+1) + kmax + ki + 1;
        y_pow = abs(his_full_est(his_shift)*cpe.pil_val)^2;
        if his_full_diff(his_shift) > 1e-13 && y_pow <= cpe.thres
            diff_num_less = diff_num_less + 1;
            fprintf("      - [%2d], origin: +%.4f, est: +%.4f\n", his_shift, his_full(his_shift), his_full_est(his_shift));
        end
    end
end
fprintf("    - CHE check (greater):\n")
diff_num_grea = 0;
for li = 0:lmax
    for ki = -kmax:kmax
        his_shift = li*(2*kmax+1) + kmax + ki + 1;
        y_pow = abs(his_full_est(his_shift)*cpe.pil_val)^2;
        if his_full_diff(his_shift) > 1e-13 && y_pow > cpe.thres
            diff_num_grea = diff_num_grea + 1;
            fprintf("      - [%2d], origin: %+.4f, est: %+.4f, diff: %.4f\n", his_shift, his_full(his_shift), his_full_est(his_shift), his_full_diff(his_shift));
        end
    end
end

if diff_num_less + diff_num_grea ~= sum(his_full_diff > 1e-13, "all")
    error("Difference not match!\n")
else
    fprintf("  - find %d difference, at max %.4f.\n", diff_num_less + diff_num_grea, max(his_full_diff, [], "all"))
end
fprintf("  - CHE full\n");
if any(abs(his_full_est1(his_mask))< 1e-13)
    error("Mask one not match!\n");
else
    fprintf("    - mask selected values are positive!\n");
end
if any(abs(his_full_est1(~his_mask)) ~= 0) 
    raise Exception("Mask zero not match!\n");
else
    fprintf("    - mask unselected values are zeros!\n");
end
if max(his_full_diff1, [], "all") > 0
    error("his not match!\n");
else
    fprintf("    - his are the same!\n");
end


