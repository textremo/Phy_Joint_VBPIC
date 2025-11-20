import numpy as np
from numpy import exp
from numpy.linalg import inv

eps = np.finfo(np.float64).eps

import torch
import torch.nn as nn


class Modu:
    # OFDM 0~49
    MODU_OFDM_STD               = 0;
    MODUS_OFDM = [MODU_OFDM_STD];
    # OTFS 50~99
    MODU_OTFS_FULL          = 50;           # full
    MODU_OTFS_EMBED         = 51;           # embed
    MODU_OTFS_SP            = 60;           # superimposed
    MODU_OTFS_SP_REP_DELAY  = 65;           # superimposed - replicate on the delay axis
    MODUS_OTFS = [MODU_OTFS_FULL, MODU_OTFS_EMBED, MODU_OTFS_SP, MODU_OTFS_SP_REP_DELAY];
    # all
    MODUs = [MODUS_OFDM, MODUS_OTFS];

    # Frame type
    FT_CP = 1;              # cyclic prefix
    FT_ZP = 2;              # zero padding
    FTs = [FT_CP, FT_ZP];

    # Pulse Type
    PUL_BIORT = 0;          # biorthogonal
    PUL_RECTA = 1;          # rectangular pulse
    PULs = [PUL_BIORT, PUL_RECTA];
    
    
    #------------------------------------------------------------------
    B = None;
    # modulation
    modu = None;
    frame = None;
    pul = None;
    N = 0;              # timeslot number
    M = 0;              # subcarrier number
    # modulation - OTFS
    K = 0;              # Doppler (timeslot) number
    L = 0;              # delay (subcarrier) number
    # pilot
    dataLocs = None;     # data locations
    refSig = None;       # reference siganl (pilot + guard)
    csiLim = None;
    # vectorized length
    sig_len = 0;
    data_len = 0;

    #------------------------------------------------------------------
    # constel
    constel = None;
    constel_len = 0;
    Ed = 1;                                             # energy of data (constellation average power)
    
    #------------------------------------------------------------------
    # CSI
    Eh = None;                                           # energy of the channel

    #------------------------------------------------------------------
    # OTFS
    # CSI
    pmax = None;
    lis = None;
    kis = None;
    # area division
    pilCheRng = None;       # pilot CHE range [k0, kN, l0, lN] (k0->kN: k range, l0-lN: l range)            
    pilCheRng_klen = 0;
    pilCheRng_len = 0;
    # others
    H0 = None;
    Hv0 = None;
    off_diag = None;
    eyeKL = None;
    eyeK = None;
    eyeL = None;
    eyePmax = None;
    hw0 = None;
    hvw0 = None;
    dftmat = None;
    idftmat = None;
    piMat = None;
    
    '''
    init
    @modu:           modulation type
    @frame:          frame type
    @pul:            pulse type
    @nTimeslot:      timeslot number
    @lmax:           the maximal delay index
    @nSubcarr:       subcarrier number
    <OPT>
    @csiLim:         CSI limitation
                     1)
                     2) OTFS: [lmax, kmax]
    '''
    def Modu(self, modu, frame, pul, nTimeslot, nSubcarr, *args, B=None):
        if modu not in self.MODUs:
            raise Exception("The modulation type is not supported!!!");
        if frame not in self.FTs:
            raise Exception("The frame type is not supported!!!");
        if pul not in self.PULs:
            raise Exception("The pulse type is not supported!!!");
        self.modu = modu;
        self.frame = frame;
        self.pul = pul;
        # dimension
        if modu < 50:
            # OFDM
            self.N = nTimeslot;
            self.M = nSubcarr;
        else:
            # OTFS
            self.K = nTimeslot;
            self.L = nSubcarr;
        self.sig_len = nTimeslot*nSubcarr;
        if len(args) >= 1:
            self.csiLim = args[0];
        if B:
            self.B = B;

        #--------------------------------------------------------------
        # OTFS properties
        if self.csiLim:
            lmax = self.csiLim[0]; kmax = self.csiLim[1];
            # delay & Doppler
            self.pmax = (lmax+1)*(2*kmax+1); 
            self.lis = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1));
            self.kis = np.arange(-kmax, kmax+1).repeat(lmax+1);
            # H0
            self.H0 = np.zeros([self.B, self.sig_len, self.sig_len]) if self.B else np.zeros([self.sig_len, self.sig_len]);
            self.Hv0 = np.zeros([self.B, self.sig_len, self.sig_len]) if self.B else np.zeros([self.sig_len, self.sig_len]);
            # off-diagonal
            self.off_diag =  np.eye(self.sig_len)+1 - np.eye(self.sig_len)*2;
            if self.B:
                self.off_diag = np.tile(self.off_diag, [self.B, 1, 1]);
            # eye
            self.eyeKL = np.eye(self.sig_len);
            self.eyeK = np.eye(self.K);
            self.eyeL = np.eye(self.L);
            self.eyePmax = np.eye(self.pmax);

            # others
            if self.pul == self.PUL_BIORT:
                # bi-orthogonal pulse
                self.hw0 = np.zeros([self.K, self.L]);
                self.hvw0 = np.zeros([self.K, self.L]);
            elif self.pul == self.PUL_RECTA:
                # rectangular pulse
                # self.dftmat = dftmtx(self.K)*sqrt(1/self.K);    # DFT matrix  
                # self.idftmat = conj(self.dftmat);               # IDFT matrix     
                # self.piMat = eye(self.sig_len);                 # permutation matrix (from the delay) -> pi
                pass
            
class Modu2(Modu, nn.Module):
    pass