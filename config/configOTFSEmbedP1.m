% otfs settings
N = 8;
M = 8;
fc = 4;                 % GHz
freq_sp = 15;           % kHz
% channel
p = 3;
lmax = 1;
kmin = -1;
kmax = 1;
% guard
gdn_len = lmax;
gdp_len = lmax;
gkn_len = 2*kmax;
gkp_len = 2*kmax;
sig_len = (N*M-(4*kmax+1)*(2*lmax+1));
% modulation
M_mod = 4;
M_bits = log2(M_mod);
constel = qammod(0: M_mod-1, M_mod, 'UnitAveragePower',true);
constel_real = unique(real(constel));
xDD_len = N*M;
% SNR ranges
SNR_ds = 10:2:18;
SNR_p = 30;
Nos = 10.^(-SNR_ds/10);
% simulation frames
N_frams = 1e3*ones(length(SNR_ds), 1);
N_frams(4:end) = 5e4;
% N_frams = 10*ones(length(SNR_ds), 1);