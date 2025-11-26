import numpy as np
from numpy import arange, ones, zeros, eye, diag, diagonal, kron, reshape, einsum, sqrt, exp, conj, real
from numpy import tile as repmat
from numpy import moveaxis as mvdim
from numpy import roll as circshift
from numpy.linalg import inv
from numpy.fft import fft, ifft
from numpy import expand_dims as unsqueeze

eps = np.finfo(np.float64).eps
try:
    from ..common.Modu import Modu
except:
    from .Modu import Modu

class VB(Modu):
    
    def __init__(self, modu, frame, pul, nTimeslot, nSubcarr, *args, B=None):
        super().__init__(modu, frame, pul, nTimeslot, nSubcarr, *args, B=B);
        
    '''
    channel estimation
    @Ydd:           Rx in the DD domain
    <OPT>
    @No:            the noise power
    @min_var:       the minimal variance.
    @iter_num:      the maximal iteration
    @es:            early stop
    @es_thres:      early stop threshold (abs)
    '''
    def che(self, Ydd, *, No=None, min_var=eps, iter_num=125, es=True, es_thres=1e-6):
        Yp = Ydd[..., self.pilCheRng[0]:self.pilCheRng[1]+1, self.pilCheRng[2]:self.pilCheRng[3]+1]
        yp = reshape(Yp, [self.B, -1, 1], order="F")
        Z = yp.shape[-2]
        P = self.ref2Phi()
        PtP = mvdim(P, -1, -2).conj() @ P
        Pty = mvdim(P, -1, -2).conj() @ yp
        a = ones([self.B, 1, 1])
        b = ones([self.B, 1, 1])
        c = ones([self.B, self.pmax, 1])
        d = ones([self.B, self.pmax, 1])
        alpha = ones([self.B, 1, 1])
        gamma = ones([self.B, self.pmax, 1])
        gamma_new = ones([self.B, self.pmax, 1])
        h_vari = inv(PtP + repmat(eye(self.pmax), [self.B, 1, 1])); 
        h_mean = h_vari @ Pty
        update_alpha = False
        if No:
            alpha = alpha/No
        else:
            update_alpha = True
            
        # VB CHE 
        upids = arange(self.B)
        for t in range(iter_num):
            # update alpha
            if update_alpha:
                a = a + Z;
                b = b + np.sum(yp - P @ h_mean, axis=(-1,-2), keepdims=True) + unsqueeze(np.sum(diagonal(PtP @ h_vari, axis1=-1, axis2=-2), axis=-1), axis=(-1,-2))
                alpha = a/b;
            
            
            # update h
            h_vari[upids] = inv(alpha[upids] * PtP[upids] + gamma[upids]*diag(ones(self.pmax)))
            h_mean[upids] = alpha[upids]* h_vari[upids] @ Pty[upids];
            # update gamma
            c[upids] = c[upids] + 1;
            d[upids] = d[upids] + real(diagonal(h_vari[upids], axis1=-2, axis2=-1)[..., None]) + abs(h_mean[upids])**2;
            gamma_new[upids] = c[upids]/d[upids];
            
            if es:
                upids = np.sum(abs(gamma_new - gamma)**2, axis=(-2,-1))/np.sum(abs(gamma)**2, axis=(-2,-1)) >= es_thres
                if sum(upids) == 0:
                    break
                upids = np.where(upids)[0]
            gamma[upids] = gamma_new[upids]
                
        return h_mean.squeeze(-1)
                