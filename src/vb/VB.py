import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy import moveaxis as mvdim
from numpy import roll as circshift
from numpy.linalg import inv
from numpy.fft import fft, ifft

eps = np.finfo(np.float64).eps

class VB(object):
    
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
        yp = reshape(Yp, [self.B, -1, 1]);
        Z = yp.shape[-2];
        P = self.ref2Phi();
        PtP = mvdim(P, -1, -2).conj() @ P;
        Pty = mvdim(P, -1, -2).conj() @ yp;
        a = 1; b = 1; c = ones([self.B, self.pmax]); d = ones([self.B, self.pmax]);
        alpha = ones([self.B]); gamma = ones(self.pmax, 1); h_vari = inv(PtP + repmat(eye(self.pmax), [self.B, 1, 1])); h_mean = h_vari@Pty;