import torch
import torch.nn as nn
from torch import arange, ones, zeros, eye, diag, diagonal, kron, reshape, where, einsum, sqrt, exp, conj
from torch import real, imag
from torch import unique
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

sm = nn.Softmax(dim=-1)

try:
    from ..vb.VB import VB
    from ..nn.GNN import GNN
except:
    from .VB import VB
    from .GNN import GNN

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
        self.register_buffer("constel_B_row", constel)
        self.register_buffer('constel_len', arr(constel.shape[0]))
        self.register_buffer('Ed', torch.sum(constel.abs()**2)/self.constel_len)    # constellation average power
        #real
        constel_r = unique(real(constel))
        self.register_buffer("constel_r", constel_r)
        self.register_buffer("constel_B_row_r", constel_r)
        self.register_buffer('constel_r_len', arr(constel_r.shape[0]))
        self.register_buffer('Ed_r', torch.sum(constel_r.abs()**2)/self.constel_r_len)    # constellation average power
    
    '''
    set the data location
    @dataLocs:       data locations, a 01 matrix of [N, M] or [K, L]
    '''
    def setDataLoc(self, dataLocs):
        dataLocs = arr(dataLocs)
        if hasattr(self, 'dataLocs'):
            self.dataLocs = None
            del self.dataLocs
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
                eyeL = eye(self.L, dtype=self.ctype)
                eyeK =  eye(self.K, dtype=self.ctype)
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
                #self.register_buffer("off_diag", repmat(eye(self.sig_len)+1 - eye(self.sig_len)*2, [self.B, 1, 1]))
                # eye
                #self.register_buffer("eyeKL",    repmat(eye(self.sig_len, dtype=self.ctype), [self.B, 1, 1]))
                #self.register_buffer("eyePmax",  repmat(eye(self.pmax, dtype=self.ctype), [self.B, 1, 1]))
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
                    dftmat = fft(eyeK)*sqrt(arr(1/self.K))   # DFT matrix
                    idftmat = conj(dftmat)                          # IDFT matrix
                    piMat = eye(self.sig_len, dtype=self.ctype)     # permutation matrix (from the delay) -> pi
                    # the T to register [pmax, B, KL, KL]
                    Ts = zeros(self.pmax, self.sig_len, self.sig_len, dtype=self.ctype)
                    for tap_id in range(self.pmax):
                        li = self.lis[tap_id].item()
                        ki = self.kis[tap_id].item()
                        # delay
                        piMati = circshift(piMat, li, 0); 
                        # Doppler
                        timeSeq = circshift(arange(-li, self.sig_len-li), -li)
                        deltaMat_diag = exp(2j*pi*ki/(self.sig_len)*timeSeq);
                        deltaMati = torch.diag_embed(deltaMat_diag)
                        # Pi, Qi, & Ti
                        #Pi = einsum('...ij,...kl->...ikjl', dftmat, eyeL).reshape(self.sig_len, self.sig_len) @ piMati 
                        Pi = kron(dftmat, eyeL) @ piMati 
                        #Qi = deltaMati @ einsum('...ij,...kl->...ikjl', idftmat, eyeL).reshape(self.B, self.sig_len, self.sig_len)
                        Qi = deltaMati @ kron(idftmat, eyeL)
                        Ti = Pi @ Qi;
                        Ts[tap_id, ...] = Ti
                self.register_buffer('Ts', Ts)
                # TtTs
                TtTs = zeros(self.pmax, self.pmax, self.sig_len, self.sig_len, dtype=self.ctype)
                for i in range(self.pmax):
                    for j in range(self.pmax):
                        TtTs[i, j, ...] = conj(Ts[i].transpose(-1,-2)) @ Ts[j] 
                self.register_buffer("TtTs", TtTs)
                # Psi
                Psis = zeros(self.pmax, self.sig_len, self.sig_len, dtype=self.ftype)
                for i in range(self.pmax):
                    li = self.lis[i].item()
                    ki = self.kis[i].item()
                    # delay
                    eyeDel = circshift(real(eyeL), li, 0)
                    # Doppler
                    eyeDop = circshift(real(eyeK), ki, 0)
                    Psis[i] = kron(eyeDop, eyeDel)
                self.register_buffer("Psis", Psis)
                    
                # for i in range(self.pmax):
                #     li = self.lis[i].item()
                #     ki = self.kis[i].item()
                #     T = Ts[i].numpy()
                #     Psi = Psis[i].numpy()
                #     res = torch.max(Psis[i] - abs(Ts[i]))
                #     print(res)
                # print()
                    
                
                
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
    set NN
    '''
    def setNN(self):
        self.gnn_che = GNN(2, 2*self.pmax,     3, self.constel_r, ntype=GNN.NTYPE_CHE)
        self.gnn_det = GNN(2, 2*self.sig_len,  3, self.constel_r, ntype=GNN.NTYPE_DET)
    
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
    @H:             the channel [(batch_size), KL, KL]
    @x:             the symbol [(batch_size), KL, 1]
    '''
    def forward(self, Y, h, hv, hm, No, *, min_var=1e-13, sym_map=False, H=None, x=None):
        Y = arr(Y).to(self.dev)
        y = reshape(Y, [self.B, self.sig_len, 1])
        h = arr(h).to(self.dev)
        hv = arr(hv).to(self.dev)
        hm = arr(hm).to(self.dev) 
        No = arr(No).to(self.dev)
        Xp = repmat(self.refSig, [self.B, 1, 1])
        xp = reshape(Xp, [self.B, self.sig_len, 1])
        min_var = arr(min_var, dtype=self.ctype).to(self.dev)
        min_var_r = real(arr(min_var/2)).to(self.dev)
        
        
        if H is None:
            che = True
        else:
            che = False
            H = arr(H).to(self.dev)
        if x is None:
            det = True
        else:
            det = False
        
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
        hm_r = cat([hm, hm], -2)
        Y_r = self.toReal(Y)
        y_r = self.toReal(y)
        xp_r = self.toReal(xp)
        
        # the estimated noise precision
        alpha = 1/(No/2)
        # the estimated channel mean and variance (the no-path location is force to small variance)
        a = zeros(self.B, 2*self.pmax, 1, dtype=self.ftype, device=self.dev)
        b = self.Eh/2*ones(self.B, 2*self.pmax, 1, dtype=self.ftype, device=self.dev)
        #a = self.toReal(h)
        #b = torch.max(hv)/2*ones(self.B, 2*self.pmax, 1, dtype=self.ftype, device=self.dev)
        
        b[~hm_r] = min_var_r
        # the estimated symbol mean and variance
        c = zeros(self.B, 2*self.sig_len, 1, dtype=self.ftype, device=self.dev)
        d = self.Ed_r*ones(self.B, 2*self.sig_len, 1, dtype=self.ftype, device=self.dev)
        
        # CSI
        if not che:
            Ht = H.transpose(-1, -2).conj()
            HtH = Ht @ H
            HtH_r = self.toReal(HtH)
            Ht_r = self.toReal(Ht)
            H_r = self.toReal(H)
        
        
        # VB structure
        for t in range(self.iter_num):
            if che:
                H, HtH = self.h2H(h, hv, hm)
                Ht = H.transpose(-1, -2).conj()
                # to real
                HtH_r = self.toReal(HtH)
                Ht_r = self.toReal(Ht)
                H_r = self.toReal(H)
            
            # BSO
            V_bso = inv(alpha*HtH_r + torch.diag_embed(1/sqz(d, -1)))
            x_bso = V_bso @ (alpha * (Ht_r @ y_r - HtH_r @ xp_r) + c/d)
            v_bso = usqz(diagonal(V_bso, dim1=-2, dim2=-1), -1)
            v_bso = v_bso.clamp(min_var_r)
            # GNN
            # GNN - generate features
            x_feat_n, x_feat_e = self.genFeatures(y_r, H_r, HtH_r, No)
            hs_feat_n = zeros([self.B, 2*self.sig_len, 64], device=self.dev)
            x_gnn_r, v_gnn_r = self.gnn_det(x_feat_n, x_feat_e, hs_feat_n, node_mean=x_bso, node_vari=v_bso, last=(t==self.iter_num-1))
            if t == self.iter_num - 1:
                break
            v_gnn_r = v_gnn_r.clamp(min_var_r)
            # to complex
            x_gnn = x_gnn_r[:, :self.sig_len, :] + 1j*x_gnn_r[:, self.sig_len:, :]
            v_gnn = v_gnn_r[:, :self.sig_len, :] + v_gnn_r[:, self.sig_len:, :]
            # DSC
            #x_dsc = x_bso
            #v_dsc = v_bso
            
            # CHE
            if che:
                # to Phi
                P, PtP = self.x2P(x_gnn, v_gnn, xp)
                Pt = P.transpose(-1, -2).conj()
                # to real
                PtP_r = self.toReal(PtP)
                Pt_r = self.toReal(Pt)
                P_r = self.toReal(P)
                # BSO
                HV_bso = inv(alpha*PtP_r + torch.diag_embed(1/sqz(b, -1)))
                h_bso = HV_bso @ (alpha * Pt_r @ y_r + a/b)
                hv_bso = usqz(diagonal(HV_bso, dim1=-2, dim2=-1), -1)
                hv_bso = hv_bso.clamp(min_var_r)
                # GNN
                h_feat_n, h_feat_e = self.genFeatures(y_r, P_r, PtP_r, No)
                hs_feat_n = zeros([self.B, 2*self.pmax, 64], device=self.dev)
                h_gnn_r, hv_gnn_r = self.gnn_che(h_feat_n, h_feat_e, hs_feat_n, node_mean=h_bso, node_vari=hv_bso)
                # to complex
                h = h_gnn_r[:, :self.pmax, :] + 1j*h_gnn_r[:, self.pmax:, :]
                hv = hv_gnn_r[:, :self.pmax, :] + hv_gnn_r[:, self.pmax:, :]
            
            # udpate
            if che:
                a = h_gnn_r
                b = hv_gnn_r
            c = x_gnn_r
            d = v_gnn_r
            
            # h_bso_c = self.toComplex(h_bso).detach().numpy()
            # hv_bso_c = (hv_bso[:, :self.pmax, :] + hv_bso[:, self.pmax:, :]).detach().numpy()
        
        if che and det:
            return H_r, x_gnn_r
        elif che:
            return H_r
        elif det:
            return x_gnn_r
            
        
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
                hmi = hm[..., i]
                if torch.any(hmi):
                    hi = h[..., i].reshape(-1, 1, 1)
                    H = H + hi.reshape(-1, 1, 1) * self.Ts[i]
                    for j in range(self.pmax):
                        hj = h[..., j].reshape(-1, 1, 1)
                        hvj = hv[..., j].reshape(-1, 1, 1)
                        hmj = hm[..., j]
                        if torch.any(hmj):
                            hij = hi.conj()*hj
                            if i==j:
                                hij = hij + hvj
                            HtH = HtH + self.TtTs[i, j]*hij
        return H, HtH
    
    '''
    transfer the estimated signal to CHE matrix
    '''
    def x2P(self, x, v, xp=None):
        if xp is not None:
            x = xp + x
            #x = xp
        P_cols = []
        Pv_cols = []
        for i in range(self.pmax):
            P_cols.append(self.Psis[i].to(self.ctype) @ x)
            Pv_cols.append(self.Psis[i] @ v)
        P = cat(P_cols, -1)
        Pv = cat(Pv_cols, -1)
        PtP = P.transpose(-1, -2).conj() @ P + torch.diag_embed(Pv.sum(-2))
        #PtP = P.transpose(-1, -2).conj() @ P
        return P, PtP
    
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
    to complex
    @in0:   a vector or a matrix [B, a, 1] or [B, a, b]
    '''
    def toComplex(self, in0):
        if in0.shape[-1] == 1:
            half_id = in0.shape[-2] // 2
            out0 = in0[..., :half_id, :] + 1j*in0[..., half_id:, :]
        else:
            row_id_half = in0.shape[-2] // 2
            col_id_half = in0.shape[-1] // 2
            
            out0 = ( in0[..., :row_id_half, :col_id_half] + in0[..., row_id_half:, col_id_half:])*0.5 + \
                   (-in0[..., :row_id_half, col_id_half:] + in0[..., row_id_half:, :col_id_half])*0.5j
            
        return out0
    
    
    '''
    generate features (y=H*x+No or y=Phi*h+No)
    @in0:       y, [(batch_size), KL, 1]
    @in1:       H or Phi, [(batch_size), KL, KL] or [(batch_size), KL, pmax]
    @in2:       HtH or PtP, [(batch_size), KL, KL] or [(batch_size), pmax, pmax]
    @in3:       the noise power, [B, 1, 1]
    '''
    def genFeatures(self, in0, in1, in2, in3):
        # node number
        n_num = in1.shape[-1]
        # node
        yTh = (in0.transpose(-1, -2) @ in1).transpose(-1, -2)
        hth_diag = usqz(diagonal(in2, dim1=-2, dim2=-1), -1)
        prec = repmat(1/in3, [1, n_num, 1])
        feat_n = cat([yTh, -hth_diag, prec], -1)
        
        # edge_mask
        mask = repmat(~eye(n_num, dtype=bool, device=self.dev), [self.B, 1, 1])
        # edge
        feat_e = cat([-in2[mask].view(self.B, n_num*(n_num-1), 1), repmat(1/in3, [1, n_num*(n_num-1), 1])], -1)
        
        
        return feat_n, feat_e
    
    
    '''
    symbol mapping (hard)
    @the probability
    '''
    def symmap(self, syms_prob):
        syms_prob = sm(syms_prob)
        syms_ids = torch.argmax(syms_prob, axis=-1, keepdim=True)
        syms_r = self.constel_r[syms_ids].detach()
        return syms_r[..., :self.sig_len, :] + syms_r[..., self.sig_len:, :]*1j
    
    #--------------------------------------------------------------------------
    # obsolete
    '''
    BSE
    @x: [B, KL, 1]
    @v: [B, KL, 1]
    '''
    def bse(self, x, v, *, min_var=5e-11):
        # BSE - Estimate P(x|y) using Gaussian distribution
        pxyPdfExpPower = -1/(2*v)*torch.square(x - self.constel_B_row_r)
        # BSE - make every row the max power is 0
        #     - max only consider the real part
        pxypdfExpNormPower = pxyPdfExpPower - pxyPdfExpPower.max(-1, keepdim=True).values
        pxyPdf = torch.exp(pxypdfExpNormPower)
        # BSE - Calculate the coefficient of every possible x to make the sum of all
        pxyPdfCoeff = 1/pxyPdf.sum(-1, keepdim=True)
        # BSE - PDF normalisation
        pxyPdfNorm = pxyPdfCoeff*pxyPdf
        # BSE - calculate the mean and variance
        x_bse_d = (pxyPdfNorm*self.constel_B_row_r).sum(-1, keepdim=True)
        v_bse_d = (torch.square(x_bse_d - self.constel_B_row_r)*pxyPdfNorm).sum(-1, keepdim=True)
        v_bse_d = v_bse_d.clamp(min_var)
        return x_bse_d, v_bse_d