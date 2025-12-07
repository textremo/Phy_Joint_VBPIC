import torch
import torch.nn as nn
from torch import arange, ones, zeros, eye, diag, diagonal, kron, reshape, where, einsum, sqrt, exp, conj
from torch import real, imag
from torch import as_tensor as arr
from torch.fft import fft, ifft
from torch import tile as repmat
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
    '''
    to
    '''
    def to(self, *args):
        super().to(*args)
        self.K = arr(self.K).to(*args)
        self.L = arr(self.L).to(*args)
        self.sig_len = arr(self.L).to(*args)
        
    '''
    constructor
    '''
    def __init__(self, modu, frame, pul, nTimeslot, nSubcarr, B=None, dev=torch.device('cpu')):
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
            
        # init
        nn.Module.__init__(self)
        VB.__init__(self, modu, frame, pul, nTimeslot, nSubcarr, B=B)
        
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
                self.register_buffer("Hv0",      zeros(self.B, self.sig_len, self.sig_len))
                # off-diagonal
                self.register_buffer("off_diag", repmat(eye(self.sig_len)+1 - eye(self.sig_len)*2, [self.B, 1, 1]))
                # eye
                self.register_buffer("eyeKL",    repmat(eye(self.sig_len, dtype=self.ctype), [self.B, 1, 1]))
                self.register_buffer("eyeK",     repmat(eye(self.K, dtype=self.ctype), [self.B, 1, 1]))
                self.register_buffer("eyeL",     repmat(eye(self.L, dtype=self.ctype), [self.B, 1, 1]))
                self.register_buffer("eyePmax",  repmat(eye(self.pmax, dtype=self.ctype), [self.B, 1, 1]))
                # others
                if self.pul == self.PUL_BIORT:
                    # bi-orthogonal pulse
                    self.register_buffer('hw0', zeros(self.B, self.K, self.L, dtype=self.ctype))
                    self.register_buffer('hvw0', zeros(self.B, self.K, self.L, dtype=self.ctype))
                elif self.pul == self.PUL_RECTA:
                    # rectangular pulse
                    # DFT matrix
                    self.register_buffer('dftmat', repmat(fft(eye(self.K))*sqrt(arr(1/self.K)), [self.B, 1, 1]))
                    # IDFT matrix
                    self.register_buffer('idftmat',conj(self.dftmat))
                    # permutation matrix (from the delay) -> pi
                    self.register_buffer('piMat', repmat(eye(self.sig_len, dtype=self.ctype), [self.B, 1, 1]))
                    
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
            refSig = refSig[:, 1:self.lmax+1];
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
    generate features (y=Hx+No, x_mean, x_vari)
    @y:         y, [(batch_size), KL, 1]
    @H:         H or Phi, [(batch_size), KL*KL] or [(batch_size), KL*pmax]
    @No:        the noise power, scalar
    @x_mean:    the estimated mean, [(batch_size), KL, 1] or [(batch_size), pmax, 1]
    @x_cova:    the estimated covariance, [(batch_size), KL, KL] or [(batch_size), pmax, pmax]
    '''
    def genFeatures(self, y, H, No, x_mean, x_cova):
        pass
    