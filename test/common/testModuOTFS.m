clear;
clc;

%{
case 01 - h2H
%}
disp("case 01 (h2H)");
nTimeslot = 6;
nSubcarr = 6;
dataLocs = ones(nTimeslot, nSubcarr); dataLocs(3:4, 3:4) = 0;
refSig = [zeros(2, nSubcarr); ...
          0, 0, -1-1j, -1+1j, 0, 0; ...
          0, 0, 1-1j, 1+1j, 0, 0; ...
          zeros(2, nSubcarr);];
lmax = 2;
kmax = 1;
csiLim = [lmax, kmax];
lis = kron(0:lmax, ones(1, 2*kmax+1));
kis = repmat(-kmax:kmax, 1, lmax+1);
%h = [0, 0, 1+2j, 0, 2+3j, 0, 0, 0, 0];h = h.';
h = randn(9,1) + randn(9,1)*1j;

moduOTFS = ModuOTFS(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, dataLocs, refSig, csiLim);
[H, Hv] = moduOTFS.h2H(h);

rg = OTFSResGrid(nSubcarr, nTimeslot);
rg.setPulse2Recta();
rg.setContent(refSig);
otfs = OTFS();
otfs.modulate(rg);
otfs.setChannel(h, lis, kis);
Hdd = otfs.getChannel();
if max(abs(H - Hdd), [], "all") <= 1e-14
    fprintf("  - Recta pass at %e.\n", max(abs(H - Hdd), [], "all"));
end

%{
case 02 - refSig
%}
disp("case 02 (refSig)");
otfs.passChannel(0);
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
Y_DD_cherng = Y_DD(moduOTFS.pilCheRng(1):moduOTFS.pilCheRng(2), moduOTFS.pilCheRng(3):moduOTFS.pilCheRng(4));
yDD_cherng = Y_DD_cherng(:);


Phi_p = moduOTFS.ref2Phi();
yDD_cherng_est = Phi_p*h;
if max(abs(yDD_cherng - yDD_cherng_est)) < 1e-14
    fprintf("  - pass at %e.\n", max(abs(yDD_cherng - yDD_cherng_est)));
end



