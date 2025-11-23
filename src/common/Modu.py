import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy import roll as circshift
from numpy.linalg import inv
from numpy.fft import fft, ifft

pi = np.pi
eps = np.finfo(np.float64).eps

import torch
import torch.nn as nn


class Modu:
    # OFDM 0~49
    MODU_OFDM_STD           = 0;
    MODUS_OFDM = [MODU_OFDM_STD];
    # OTFS 50~99
    MODU_OTFS_FULL          = 50;           # full
    MODU_OTFS_EMBED         = 51;           # embed
    MODU_OTFS_SP            = 60;           # superimposed
    MODU_OTFS_SP_REP_DELAY  = 65;           # superimposed - replicate on the delay axis
    MODUS_OTFS = [MODU_OTFS_FULL, MODU_OTFS_EMBED, MODU_OTFS_SP, MODU_OTFS_SP_REP_DELAY];
    # all
    MODUs = MODUS_OFDM + MODUS_OTFS

    # Frame type
    FT_CP = 1;              # cyclic prefix
    FT_ZP = 2;              # zero padding
    FTs = [FT_CP, FT_ZP];

    # Pulse Type
    PUL_BIORT = 0;          # biorthogonal
    PUL_RECTA = 1;          # rectangular pulse
    PULs = [PUL_BIORT, PUL_RECTA];
    
    
    #------------------------------------------------------------------
    B = 1;
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
    def __init__(self, modu, frame, pul, nTimeslot, nSubcarr, *args, B=None):
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
        self.init();
        
    '''
    init
    '''
    def init(self):
        if self.csiLim:
            lmax = self.csiLim[0]; kmax = self.csiLim[1];
            # delay & Doppler
            self.pmax = (lmax+1)*(2*kmax+1); 
            self.lis = kron(arange(lmax+1), ones(2*kmax + 1)).astype(int);
            self.kis = arange(-kmax, kmax+1).repeat(lmax+1).astype(int);
            # H0
            self.H0 = zeros([self.B, self.sig_len, self.sig_len]);
            self.Hv0 = zeros([self.B, self.sig_len, self.sig_len]);
            # off-diagonal
            self.off_diag =  repmat(eye(self.sig_len)+1 - eye(self.sig_len)*2, [self.B, 1, 1]);
            # eye
            self.eyeKL = repmat(eye(self.sig_len), [self.B, 1, 1]);
            self.eyeK = repmat(eye(self.K), [self.B, 1, 1]);
            self.eyeL = repmat(eye(self.L), [self.B, 1, 1]);
            self.eyePmax = repmat(eye(self.pmax), [self.B, 1, 1]);

            # others
            if self.pul == self.PUL_BIORT:
                # bi-orthogonal pulse
                self.hw0 = zeros([self.B, self.K, self.L]);
                self.hvw0 = zeros([self.B, self.K, self.L]);
            elif self.pul == self.PUL_RECTA:
                # rectangular pulse
                self.dftmat = repmat(fft(eye(self.K)), [self.B, 1, 1])*sqrt(1/self.K);      # DFT matrix  
                self.idftmat = conj(self.dftmat);                                           # IDFT matrix     
                self.piMat = repmat(eye(self.sig_len), [self.B, 1, 1]);                     # permutation matrix (from the delay) -> pi
                
                
    '''
    set the data location
    @dataLocs:       data locations, a 01 matrix of [N, M] or [K, L]
    '''
    def setDataLoc(self, dataLocs):
        self.dataLocs = dataLocs;
        self.data_len = np.sum(dataLocs);

    '''
    set the reference signal
    @refSig:         the reference sigal of [N, M] or [K, L], 0 at non-ref locations
    '''
    def setRef(self, refSig):
        refSig = np.asarray(refSig);
        if refSig.ndim > 2:
            raise Exception("The reference signal cannot be batched!!!");
        self.refSig = refSig;
        if self.modu == self.MODU_OTFS_FULL:
            raise Exception("Full data does not support any reference signal!!!");
        
        # pilot CHE range
        if self.csiLim:
            lmax = self.csiLim[0]; kmax = self.csiLim[1];
            refSig = self.refSig;
            if self.modu == self.MODU_OTFS_SP_REP_DELAY:
                refSig = self.refSig[:, 1:lmax+1];
            pkls = np.asarray(np.where(abs(refSig) > eps))
            pk0 = pkls[0, 0]
            pl0 = pkls[1, 0]
            pkN = pkls[0, -1]
            plN = pkls[1, -1]
            
            self.pilCheRng = [max(pk0-kmax, 0), min(pkN+kmax, self.K-1), pl0, min(plN + lmax, self.L-1)];
            self.pilCheRng_klen = self.pilCheRng[1] - self.pilCheRng[0] + 1;
            self.pilCheRng_len = self.pilCheRng_klen*(self.pilCheRng[3] - self.pilCheRng[2] + 1);
            
    '''
    set the constellation
    @constel:           the constellation, a vector
    '''
    def setConstel(self, constel):
        constel = np.asarray(constel)
        
        self.constel = constel                              # constellation must be a row vector or an 1D vector
        self.constel_len = len(constel);
        self.Ed = sum(abs(constel)**2)/self.constel_len;    # constellation average power
    
    '''
    set the csi if know
    @Eh:                the energy of each path
    '''
    def setCSI(self, Eh):
        self.Eh = Eh;
        
    '''
    check the pulse type
    '''
    def isPulBiort(self):
        return self.pul == self.PUL_BIORT;
    def isPulRecta(self):
        return self.pul == self.PUL_RECTA;
    
    
    '''
    h to H (time domain to DD domain)
    @h:        CHE path gains, [B, Pmax]
    <OPT>
    @hv:       CHE variance, [B, Pmax]
    @hm:       CHE mask, [B, Pmax]
    @min_var:  the minimal variance
    '''
    def h2H(self, h, *, hv=None, hm=None, min_var=eps):
        if not hv:
            hv = ones([self.B, self.pmax]);
        if not hm:
            hm = ones([self.B, self.pmax]);
        
        # to H
        H = self.H0;
        Hv = self.Hv0;
        if self.pul == self.PUL_BIORT:
            pass
        if self.pul == self.PUL_RECTA:
            for tap_id in range(self.pmax):
                hmi = hm[..., tap_id];
                # only accumulate when there are at least a path
                if np.any(hmi):
                    hi = h[..., tap_id]
                    hvi = hv[..., tap_id]
                    li = self.lis[tap_id].item()
                    ki = self.kis[tap_id].item()
                    # delay
                    piMati = circshift(self.piMat, li, 1); 
                    # Doppler
                    timeSeq = repmat(circshift(arange(-li, self.sig_len-li), -li), [self.B, 1])
                    deltaMat_diag = exp(2j*pi*ki/(self.sig_len)*timeSeq);
                    deltaMati = deltaMat_diag[..., None]*eye(self.sig_len)
                    # Pi, Qi, & Ti
                    Pi = einsum('...ij,...kl->...ikjl', self.dftmat, self.eyeL).reshape(self.B, self.sig_len, self.sig_len) @ piMati 
                    Qi = deltaMati @ einsum('...ij,...kl->...ikjl', self.idftmat, self.eyeL).reshape(self.B, self.sig_len, self.sig_len)
                    Ti = Pi @ Qi;
                    H = H + hi.reshape(-1, 1, 1) * Ti;
                    Hv = Hv + hvi.reshape(-1, 1, 1)*abs(Ti);
 
        # set the minimal variance
        Hv = Hv.clip(min_var)
        return H, Hv
    
    '''
    refSig to Phi
    '''
    def ref2Phi(self):
        if self.modu == self.MODU_OTFS_FULL:
            raise Exception("Not refence signal is given on the full data frame type!!!");
        
        Phi = zeros([self.B, self.pilCheRng_len, self.pmax]).astype(complex);
        for yk in arange(self.pilCheRng[0], self.pilCheRng[1]+1):
            for yl in arange(self.pilCheRng[2], self.pilCheRng[3]+1):
                Phi_ri = (yl - self.pilCheRng[2])*self.pilCheRng_klen + yk - self.pilCheRng[0];
                for p_id in arange(self.pmax):
                    li = self.lis[p_id];
                    ki = self.kis[p_id];
                    # x(k, l)
                    xl = yl - li;
                    xk = yk - ki;
                    if abs(self.refSig[xk, xl]) > eps:
                        # exponential part (pss_beta)
                        if self.isPulBiort():
                            pss_beta = exp(-2j*pi*li/self.L*ki/self.K);
                        elif self.isPulRecta():
                            pss_beta = exp(2j*pi*(yl-li)/self.L*ki/self.K);     # here, you must use `yl-li` instead of `xl` or there will be an error
                        Phi[..., Phi_ri, p_id] = self.refSig[xk, xl]*pss_beta;
        return Phi