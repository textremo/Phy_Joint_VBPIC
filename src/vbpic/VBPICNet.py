import torch
import torch.nn as nn
from torch import arange, ones, zeros, eye, diag, diagonal, kron, reshape, where, einsum, sqrt, exp, conj
from torch import real, imag
from torch import as_tensor as arr
from torch.fft import fft, ifft
from torch import tile as repmat
from torch import roll as circshift
from torch.linalg import inv
from torch import squeeze as sqz
from torch import unsqueeze as usqz
from torch import cat

from textremo_toolbox import *

eps = torch.finfo().eps
pi = torch.pi

try:
    from ..vb.VB import VB
except:
    from .VB import VB

class VBPICNet(VB, nn.Module):
    NN_TYPE_MEANVARI = 1;       # use mean and variance as extra features
    NN_TYPE_MEANCOVA = 2;       # use mean and covariance as extra features
    
    
    '''
    to
    '''
    def to(self, *args):
        super().to(*args)
        self.K = arr(self.K).to(*args)
        self.L = arr(self.L).to(*args)
        self.sig_len = arr(self.sig_len).to(*args)
        
    '''
    constructor
    '''
    def __init__(self, modu, frame, pul, nTimeslot, nSubcarr, B=None, dev=torch.device('cpu'), *, iter_num=10):
        # data precision
        self.ftype = torch.get_default_dtype()
        if self.ftype == torch.float16:
            self.itype = torch.int16
            self.ctype = torch.complex32
        elif self.ftype == torch.float32:
            self.itype = torch.int32
            self.ctype = torch.complex64
        elif self.ftype == torch.float64:
            self.itype = torch.int64
            self.ctype = torch.complex128
        # device
        self.dev = dev
        # iter_num
        self.iter_num = iter_num
            
        # init
        nn.Module.__init__(self)
        VB.__init__(self, modu, frame, pul, nTimeslot, nSubcarr, B=B)
        # dataLoc
        self.register_buffer('dataLocs', ones(B, self.sig_len, self.sig_len))
        
    '''
    set constel
    '''
    def setConstel(self, constel):
        constel = sqz(arr(constel))
        self.register_buffer("constel", constel)
        self.register_buffer("constel_B_row", repmat(constel, [self.B, 1, 1]))
        self.register_buffer('constel_len', arr(constel.shape[0]))
        self.register_buffer('Ed', torch.sum(constel.abs()**2)/self.constel_len)    # constellation average power
    
    '''
    set the data location
    @dataLocs:       data locations, a 01 matrix of [N, M] or [K, L]
    '''
    def setDataLoc(self, dataLocs):
        dataLocs = arr(dataLocs)
        self.register_buffer('dataLocs', dataLocs)
        self.data_len = torch.sum(dataLocs)
        
    '''
    set the csi if know
    <OTFS>
    @ins: Eh, p, kmax,lmax; Eh, kmax,lmax; kmax, lmax
    '''
    def setCSI(self, *args):
        if self.modu in self.MODUS_OTFS:
            if len(args) < 2:
                raise Exception("The CSI information is incomplete!!!")
            else:
                kmax = args[-2]
                lmax = args[-1]
                eyeL = repmat(eye(self.L, dtype=self.ctype), [self.B, 1, 1])
                
                self.register_buffer('kmax',     arr(kmax))
                self.register_buffer('lmax',     arr(lmax))
                if len(args) >= 3:
                    self.register_buffer('Eh',   arr(args[0]))
                if len(args) == 4:
                    self.register_buffer('p',    arr(args[1]))
                self.pmax = (lmax+1)*(2*kmax+1) 
                self.register_buffer("lis",      kron(arange(lmax+1), ones(2*kmax + 1, dtype=self.itype)))
                self.register_buffer("kis",      repmat(arange(-kmax, kmax+1), [lmax+1]))
                # H0
                self.register_buffer("H0",       zeros(self.B, self.sig_len, self.sig_len, dtype=self.ctype))
                self.register_buffer("HtH0",     zeros(self.B, self.sig_len, self.sig_len, dtype=self.ctype))
                # off-diagonal
                self.register_buffer("off_diag", repmat(eye(self.sig_len)+1 - eye(self.sig_len)*2, [self.B, 1, 1]))
                # eye
                self.register_buffer("eyeKL",    repmat(eye(self.sig_len, dtype=self.ctype), [self.B, 1, 1]))
                self.register_buffer("eyePmax",  repmat(eye(self.pmax, dtype=self.ctype), [self.B, 1, 1]))
                # others
                if self.pul == self.PUL_BIORT:
                    # bi-orthogonal pulse
                    self.register_buffer('hw0', zeros(self.B, self.K, self.L, dtype=self.ctype))
                    self.register_buffer('hvw0', zeros(self.B, self.K, self.L, dtype=self.ctype))
                # Ts
                if self.pul == self.PUL_BIORT:
                    pass
                if self.pul == self.PUL_RECTA:
                    # tmp variables
                    dftmat = repmat(fft(eye(self.K))*sqrt(arr(1/self.K)), [self.B, 1, 1])   # DFT matrix
                    idftmat = conj(dftmat)                                                  # IDFT matrix
                    piMat = repmat(eye(self.sig_len, dtype=self.ctype), [self.B, 1, 1])     # permutation matrix (from the delay) -> pi
                    # the T to register [pmax, B, KL, KL]
                    Ts = zeros(self.pmax, self.B, self.sig_len, self.sig_len, dtype=self.ctype)
                    for tap_id in range(self.pmax):
                        li = self.lis[tap_id].item()
                        ki = self.kis[tap_id].item()
                        # delay
                        piMati = circshift(piMat, li, 1); 
                        # Doppler
                        timeSeq = repmat(circshift(arange(-li, self.sig_len-li), -li), [self.B, 1])
                        deltaMat_diag = exp(2j*pi*ki/(self.sig_len)*timeSeq);
                        deltaMati = torch.diag_embed(deltaMat_diag)
                        # Pi, Qi, & Ti
                        Pi = einsum('...ij,...kl->...ikjl', dftmat, eyeL).reshape(self.B, self.sig_len, self.sig_len) @ piMati 
                        Qi = deltaMati @ einsum('...ij,...kl->...ikjl', idftmat, eyeL).reshape(self.B, self.sig_len, self.sig_len)
                        Ti = Pi @ Qi;
                        Ts[tap_id, ...] = Ti
                self.register_buffer('Ts', Ts)
                # TtTs
                TtTs = zeros(self.pmax, self.pmax, self.B, self.sig_len, self.sig_len, dtype=self.ctype)
                for i in range(self.pmax):
                    for j in range(self.pmax):
                        TtTs[i, j, ...] = Ts[i] @ Ts[j] 
                self.register_buffer("TtTs", TtTs)
                
    '''
    set the reference signal
    @refSig:         the reference sigal of [N, M] or [K, L], 0 at non-ref locations
    '''
    def setRef(self, refSig):
        refSig = arr(refSig);
        if refSig.ndim > 2:
            raise Exception("The reference signal cannot be batched!!!");
        self.register_buffer('refSig', refSig)
        self.register_buffer('refSigN', repmat(refSig, [self.B, 1, 1]))
        if self.modu == self.MODU_OTFS_FULL:
            raise Exception("Full data does not support any reference signal!!!");
        
        # pilot CHE range
        if self.modu == self.MODU_OTFS_SP_REP_DELAY:
            refSig = refSig[:, 0:self.lmax+1]
        pks, pls = where(refSig.abs() > eps)
        pk0 = pks[0]
        pl0 = pls[0]
        pkN = pks[-1]
        plN = pls[-1]
        self.register_buffer('pilCheRng', 
                             arr([max(pk0-self.kmax, 0),  min(pkN+self.kmax, self.K-1), 
                                  pl0,                    min(plN + self.lmax, self.L-1)]))
        self.register_buffer('pilCheRng_klen', self.pilCheRng[1] - self.pilCheRng[0] + 1)
        self.register_buffer('pilCheRng_len', self.pilCheRng_klen*(self.pilCheRng[3] - self.pilCheRng[2] + 1))
        
    '''
    refSig to Phi
    '''
    def ref2Phi(self):
        if self.modu == self.MODU_OTFS_FULL:
            raise Exception("Not refence signal is given on the full data frame type!!!");
        Phi = []
        for yl in arange(self.pilCheRng[2], self.pilCheRng[3]+1).to(self.dev):
            for yk in arange(self.pilCheRng[0], self.pilCheRng[1]+1).to(self.dev):      
                Phi_r = []
                for p_id in arange(self.pmax):
                    li = self.lis[p_id]
                    ki = self.kis[p_id]
                    # x(k, l)
                    xl = yl - li
                    xk = yk - ki
                    # exponential part (pss_beta)
                    if self.isPulBiort():
                        pss_beta = exp(-2j*pi*li/self.L*ki/self.K);
                    elif self.isPulRecta():
                        pss_beta = exp(2j*pi*(yl-li)/self.L*ki/self.K);     # here, you must use `yl-li` instead of `xl` or there will be an error
                    Phi_r.append(self.refSigN[..., xk, xl][..., None, None]*pss_beta)
                Phi.append(cat(Phi_r, -1))
        return cat(Phi, -2)
    
    '''
    channel estimation
    @Ydd:           Rx in the DD domain
    <OPT>
    @No:            the noise power
    @min_var:       the minimal variance.
    @iter_num:      the maximal iteration
    @es:            early stop
    @es_thres:      early stop threshold (abs)
    @nn:            use neural network or not
    '''
    def che(self, Ydd, *, No=None, min_var=eps, iter_num=125, es=True, es_thres=1e-6, nn=True):
        Ydd = arr(Ydd).to(self.dev)
        if No:
            No = arr(No).to(self.dev)
            update_alpha = False
        else:
            No = arr(1).to(self.dev)
            update_alpha = True
        min_var = arr(min_var).to(self.dev)
        es_thres = arr(es_thres).to(self.dev)
        # base variables
        Yp = Ydd[..., self.pilCheRng[0]:self.pilCheRng[1]+1, self.pilCheRng[2]:self.pilCheRng[3]+1]
        yp = Yp.movedim(-2, -1).contiguous().view(self.B, -1, 1)
        P = self.ref2Phi()
        # extra variables
        if not nn:
            Z = yp.shape[-2]
            PtP = P.movedim(-2, -1).contiguous().conj() @ P
            Pty = P.movedim(-2, -1).conj() @ yp
            a = ones(self.B, 1, 1, dtype=self.ftype, device=self.dev)
            b = ones(self.B, 1, 1, dtype=self.ftype, device=self.dev)
            c = ones(self.B, self.pmax, 1, dtype=self.ftype, device=self.dev)
            d = ones(self.B, self.pmax, 1, dtype=self.ftype, device=self.dev)
            alpha = repmat(1/No, [self.B,1,1])
            gamma = ones(self.B, self.pmax, 1, dtype=self.ftype, device=self.dev)
            gamma_new = ones(self.B, self.pmax, 1, dtype=self.ftype, device=self.dev)
            I = diag(ones(self.pmax, device=self.dev))
            h_vari = inv(PtP + repmat(eye(self.pmax, dtype=self.ftype, device=self.dev), [self.B, 1, 1])) 
            h_mean = h_vari @ Pty
            
        else:
            yp = cat([real(yp), imag(yp)], -2)
            Z = yp.shape[-2]
            P = cat([cat([real(P), -imag(P)], -1), cat([imag(P), real(P)], -1)], -2)
            PtP = P.movedim(-2, -1).contiguous().conj() @ P
            Pty = P.movedim(-2, -1).conj() @ yp
            a = ones(self.B, 1, 1, dtype=self.ftype, device=self.dev)
            b = repmat(arr(2, dtype=self.ftype, device=self.dev), [self.B, 1, 1])
            c = ones(self.B, 2*self.pmax, 1, dtype=self.ftype, device=self.dev)
            d = repmat(arr(2, dtype=self.ftype, device=self.dev), [self.B, 2*self.pmax, 1, 1])
            I = diag(ones(2*self.pmax, device=self.dev))
            h_vari = inv(PtP + I) 
            h_mean = h_vari @ Pty
            
            
            
        # VB CHE 
        upids = arange(self.B, dtype=self.itype).to(self.dev)
        for t in range(iter_num):
            # update alpha
            if update_alpha:
                a = a + Z;
                b = b + torch.sum(yp - P @ h_mean, dim=(-1,-2), keepdims=True) + \
                    torch.sum(diagonal(PtP @ h_vari, dim1=-1, dim2=-2), -1).view(-1, 1, 1)
                alpha = a/b;
            
            
            # update h
            h_vari[upids] = inv(alpha[upids] * PtP[upids] + gamma[upids]*I)
            h_mean[upids] = alpha[upids]* h_vari[upids] @ Pty[upids];
            # update gamma
            c[upids] = c[upids] + 1;
            d[upids] = d[upids] + real(diagonal(h_vari[upids], dim1=-2, dim2=-1)[..., None]) + h_mean[upids].abs()**2;
            gamma_new[upids] = c[upids]/d[upids];
            
            if es:
                upids = torch.sum(abs(gamma_new - gamma)**2, axis=(-2,-1))/torch.sum(gamma.abs()**2, axis=(-2,-1)) >= es_thres
                if torch.sum(upids) == 0:
                    break
                upids = where(upids)[0]
            gamma[upids] = gamma_new[upids]
                
        return h_mean.squeeze(-1)
    
    '''
    detect
    @Y:             OFDM/OTFS frame [(batch_size), M, N] or [(batch_size), K, L]
    @h:             initial channel estimation - path gains [B, Pmax]
    @hv:            initial channel estimation - variance   [B, Pmax]
    @hm:            initial channel estimation - mask       [B, Pmax]
    @No:            the noise power
    <opt>
    @min_var:       the minimal variance 1e-10 by default
    @sym_map:       false by default. If true, the output will be mapped to the constellation
    '''
    def detect(self, Y, h, hv, hm, No, *, min_var=1e-10, sym_map=False):
        Y = arr(Y).to(self.dev)
        h = arr(h).to(self.dev)
        hv = arr(hv).to(self.dev)
        hm = arr(hm).to(self.dev) 
        No = arr(No).to(self.dev)
        Xp = self.refSig
        xp = reshape(xp, [self.B, self.sig_len, 1])
        
        if self.modu in self.MODUS_OFDM:
            if Y.shape[0]!= self.B or Y.shape[-2] != self.M or Y.shape[-1] != self.N:
                raise Exception("The received frame does is not the correct shape!!!")
        if self.modu in self.MODUS_OTFS:
            if Y.shape[0]!= self.B or Y.shape[-2] != self.K or Y.shape[-1] != self.L:
                raise Exception("The received frame does is not the correct shape!!!")
        #TODO: h, hv, hm check
        if h.ndim <= 2:
            h = h[..., None]
        if hv.ndim <= 2:
            hv = hv[..., None]
        if hm.ndim <= 2:
            hm = hm[..., None]
        if No.ndim == 0:
            No = repmat(No, [self.B, 1, 1])
        elif No.ndim == 1:
            No = No[..., None, None]
        else:
            raise Exception("The noise power is not in the correct shape!!!")
        # to real
        Y = self.toReal(Y)
        
        for t in range(self.iter_num):
            H, HtH = self.h2H(h, hv, hm)
            Ht = H.transpose(-1, -2).conj()
            Hty = Ht @ y
            x_bso = torch.linalg.solve(
                HtH,
                Hty - HtH @ xp
                )
            
    '''
    BSE
    '''
    def bse(self, x, v):
        pass
            
        
    #--------------------------------------------------------------------------
    # OTFS functions
    '''
    transfer the channel from time domain to the delay Doppler domain
    '''
    def h2H(self, h, hv, hm):
        # remove the last dimension (if it is 1)
        h = h.squeeze(-1)
        hv = hv.squeeze(-1)
        hm = hm.squeeze(-1)
        
        # to H
        H = self.H0
        HtH = self.HtH0
        if self.pul == self.PUL_BIORT:
            pass
        if self.pul == self.PUL_RECTA:
            for i in range(self.pmax):
                hi = conj(h[..., i]).reshape(-1, 1, 1)
                hmi = hm[..., i]
                if torch.any(hmi):
                    hi = h[..., i]
                    H = H + hi.reshape(-1, 1, 1) * self.Ts[i]
                    for j in range(self.pmax):
                        hj = h[..., j].reshape(-1, 1, 1)
                        hvj = hv[..., j].reshape(-1, 1, 1)
                        hmj = hm[..., j]
                        if torch.any(hmj):
                            hij = hi*hj + hvj if i==j else hi*hj
                            HtH = HtH + self.TtTs[i, j]*hij
        return H, HtH
    
    #--------------------------------------------------------------------------
    # AI related functions
    '''
    to real
    @in0:   a vector or a matrix [B, a, 1] or [B, a, b]
    '''
    def toReal(self, in0):
        if in0.shape[-1] == 1:
            out0 = cat([real(in0), imag(in0)], -2)
        else:
            out0 = cat([cat([real(in0), -imag(in0)], -1), cat([imag(in0), real(in0)], -1)], -2)
        return out0
    
    
    '''
    generate features (y=H*x+No or y=Phi*h+No)
    @in0:       y, [(batch_size), KL, 1]
    @in1:       H or Phi, [(batch_size), KL*KL] or [(batch_size), KL*pmax]
    @in2:       the noise power, scalar
    @in3:       the estimated mean, [(batch_size), KL, 1] or [(batch_size), pmax, 1]
    @in4:       (1) the estimated variance, [(batch_size), KL, 1] or [(batch_size), pmax, 1]
                (2) the estimated covariance, [(batch_size), KL, KL] or [(batch_size), pmax, pmax]
    '''
    def genFeatures(self, in0, in1, in2, in3, in4):
        # input check
        pass
        