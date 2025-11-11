clear;
clc;

%{
case 02 - h2H
%}
disp("case 01 (h2H)");
nTimeslot = 4;
nSubcarr = 4;
dataLocs = [1, 1, 1, 1; 1, 0, 0, 1; 1, 0, 0, 1; 1, 1, 1, 1];
refSig = [0, 0, 0, 0; 0, -1+1j, -1-1j, 0; 0, 1+1j, 1-1j, 0; 0, 0, 0, 0];
lmax = 2;
kmax = 1;
csiLim = [lmax, kmax];
lis = kron(0:lmax, ones(1, 2*kmax+1));
kis = repmat(-kmax:kmax, 1, lmax+1);
h = [0, 0, 1, 0, 2, 0, 0, 0, 0];
moduOTFS = ModuOTFS(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, dataLocs, refSig, csiLim);
[H, Hv] = moduOTFS.h2H(h);

rg = OTFSResGrid(nSubcarr, nTimeslot);
rg.setPulse2Recta();
otfs = OTFS();
otfs.modulate(rg);
otfs.setChannel(h, lis, kis);
Hdd = otfs.getChannel();
if max(abs(H - Hdd), [], "all") <= eps
    disp("  - Recta pass.");
end
