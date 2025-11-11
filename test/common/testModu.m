clear;
clc;

disp("Modu");
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

try
    modu = Modu(Modu.MODU_OTFS_FULL, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, dataLocs, refSig);
catch
    disp("  - illegal refSig pass.");
end
modu = Modu(Modu.MODU_OTFS_FULL, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, dataLocs);
modu = Modu(Modu.MODU_OTFS_EMBED, Modu.FT_CP, Modu.PUL_RECTA, nTimeslot, nSubcarr, dataLocs, refSig, csiLim);
disp("  - all pass.");