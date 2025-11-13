import torch
import torch.nn as nn

from whatshow_toolbox import *;

eps = torch.finfo().eps 

class JPIC(nn.Module):
    # constants
    # CE (channel estimation)
    CE_MRC = 1;
    CE_ZF = 1;
    # SD (symbol detection)
    # SD - BSO
    # SD - BSO - mean
    # SD - BSO - mean - 1st iter
    SD_BSO_MEAN_CAL_INIT_MMSE   = 1;
    SD_BSO_MEAN_CAL_INIT_MRC    = 2;
    SD_BSO_MEAN_CAL_INIT_LS     = 3;
    # SD - BSO - mean - other iters
    SD_BSO_MEAN_CAL_MRC = 1;    
    SD_BSO_MEAN_CAL_LS  = 2;
    # SD - BSO - var
    SD_BSO_VAR_TYPE_APPRO = 1;
    SD_BSO_VAR_TYPE_ACCUR = 2;
    SD_BSO_VAR_CAL_MMSE   = 1;
    SD_BSO_VAR_CAL_MRC    = 2;  
    SD_BSO_VAR_CAL_LS     = 3;
    # SD - DSC
    # SD - DSC - instantaneous square error
    SD_DSC_ISE_MMSE    = 1;
    SD_DSC_ISE_MRC     = 2;
    SD_DSC_ISE_LS      = 3;
    # SD - OUT
    SD_OUT_BSE = 1;
    SD_OUT_DSC = 2;
    # confidence
    CONF_RECIP_MIN = 1e-10;
    CONF_RECIP = 1e-6;
    # batch
    B0 = None;
    
    # properties
    constel = None;
    constel_len = 0;
    Ed = 1;                             # energy of data (constellation average power)
    Eh = 1;
    # CE
    ce_type                 = CE_MRC;
    # SD
    sd_bso_mean_cal_init    = SD_BSO_MEAN_CAL_INIT_MMSE;
    sd_bso_mean_cal         = SD_BSO_MEAN_CAL_MRC;
    sd_bso_var              = SD_BSO_VAR_TYPE_APPRO;
    sd_bso_var_cal          = SD_BSO_VAR_CAL_MRC;
    sd_dsc_ise              = SD_DSC_ISE_MRC;
    sd_out                  = SD_OUT_DSC;
    # OTFS configuration
    oc = None
    Xp = None                           # pilots values (a matrix)
    XpMap = None                        # the pilot map
    # batch
    B = B0;
    # control
    min_var                 = eps;      # the default minimal variance is 2.2204e-16
    iter_num                = 10;       # maximal iteration
    # control - JPIC
    es                      = False;    # early stop
    es_thres                = eps;      # early stop threshold (abs)
    # control - JPICNet
    is_net                  = False;    # using the net or not
    device = torch.device('cpu')
    
    
    '''
    constructor
    @oc:                    OTFS configuration
    @constel:               the constellation, a vector
    @lmax:                  the maximal delay index
    @kmin:                  the minimal Doppler index
    @kmax:                  the maximal Doppler index
    @B(opt):                batch size
    @min_var(opt):          the minimal variance.
    @iter_num(opt):         the maximal iteration
    @es(opt):               early stop
    @es_thres(opt):         early stop threshold (abs)
    @is_net(opt):           using the net or not
    @device(opt):           the device to run the model
    '''
    def __init__(self, oc, constel, lmax, kmin, kmax, *, B=None, min_var=None, iter_num=None, es=None, es_thres=None, is_net=None, device=None):
        super().__init__()
        # inputs
        self.oc = oc;
        constel = torch.as_tensor(constel).squeeze();
        if constel.ndim != 1:
            raise Exception("The constellation must be a vector.");
        else:
            self.constel = constel;
            self.constel_len = len(constel);
            self.Ed = torch.sum(abs(constel)**2)/self.constel_len;
        # optionl inputs
        if B is not None:           self.B          = B
        if min_var is not None:     self.min_var    = min_var
        if iter_num is not None:    self.iter_num   = iter_num
        if es is not None:          self.es         = es
        if es_thres is not None:    self.es_thres   = es_thres
        if is_net is not None:      self.is_net     = is_net
        if device is not None:      self.device     = device
        
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
            
        # buffer
        # constellation
        self.register_buffer("constel_B_row", self.constel.repeat(self.B, 1, 1))
        # delay & Doppler
        self.pmax = (lmax+1)*(2*kmax+1) 
        self.register_buffer("lis", torch.kron(torch.arange(lmax+1), torch.ones(kmax - kmin + 1, dtype=int)))
        self.register_buffer("kis", torch.arange(kmin, kmax+1).repeat(lmax+1))
        # H0
        self.register_buffer("H0", torch.zeros(self.B, self.oc.sig_len, self.oc.sig_len, dtype=self.ctype))
        self.register_buffer("Hv0", torch.zeros(self.B, self.oc.sig_len, self.oc.sig_len))
        # off-diagonal
        self.register_buffer("off_diag", ((torch.eye(self.oc.sig_len)+1) - torch.eye(self.oc.sig_len)*2).repeat(self.B, 1, 1))
        # eye
        self.register_buffer("eyeKL", torch.eye(self.oc.sig_len, dtype=self.ctype).repeat(self.B, 1, 1))
        self.register_buffer("eyeK", torch.eye(self.oc.K, dtype=self.ctype).repeat(self.B, 1, 1))
        self.register_buffer("eyeL", torch.eye(self.oc.L, dtype=self.ctype).repeat(self.B, 1, 1))
        self.register_buffer("eyePmax", torch.eye(self.pmax, dtype=self.ctype).repeat(self.B, 1, 1))
        # rectangular pulse
        if self.oc.isPulRecta():
            # DFT matrix  
            self.register_buffer("dftmat", torch.fft.fft(torch.eye(self.oc.K).repeat(self.B, 1, 1))*torch.sqrt(torch.as_tensor(1/self.oc.K)))
            # IDFT matrix         
            self.register_buffer("idftmat", torch.conj(self.dftmat))       
            # permutation matrix (from the delay) -> pi        
            self.register_buffer("piMat", torch.eye(self.oc.sig_len, dtype=self.ctype).repeat(self.B, 1, 1))            
        # Symbol Detection
        self.register_buffer("x_bse0", torch.zeros(self.B, self.oc.sig_len, 1, dtype=self.ctype))
        self.register_buffer("v_bse0", torch.zeros(self.B, self.oc.sig_len, 1))
        # channel estimation
        self.register_buffer("Phi0", torch.zeros(self.B, self.oc.K, self.oc.L, dtype=self.ctype))
        
        # nn.parameters
        
        
    '''
    settings - CE
    '''
    def setCE2MRC(self):
        self.ce_type = self.CE_MRC;
    def setCE2ZF(self):
        self.ce_type = self.CE_ZF;
        
    '''
    settings - SD
    '''
    # settings - SD - BSO - mean - cal (init)
    def setSdBsoMealCalInit2MMSE(self):
        self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MMSE;
    def setSdBsoMealCalInit2MRC(self):
        self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MRC;
    def setSdBsoMealCalInit2LS(self):
        self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_LS;
    # settings - SD - BSO - mean - cal
    def setSdBsoMeanCal2MRC(self):
        self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_MRC;    
    def setSdBsoMeanCal2LS(self):
        self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_LS;
    # settings - SD - BSO - var
    # def setSdBsoVar2Appro(self):
    #     self.sd_bso_var = self.SD_BSO_VAR_TYPE_APPRO;
    # def setSdBsoVar2Accur(self):
    #     self.sd_bso_var = self.SD_BSO_VAR_TYPE_ACCUR;
    # settings - SD - BSO - var - cal
    # def setSdBsoVarCal2MMSE(self):
    #     self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_MMSE;
    # def setSdBsoVarCal2MRC(self):
    #     self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_MRC;
    # def setSdBsoVarCal2LS(self):
    #     self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_LS;
    # settings - SD - DSC - instantaneous square error
    # def setSdDscIse2MMSE(self):
    #     self.sd_dsc_ise = self.SD_DSC_ISE_MMSE;
    # def setSdDscIse2MRC(self):
    #     self.sd_dsc_ise = self.SD_DSC_ISE_MRC;
    # def setSdDscIse2LS(self):
    #     self.sd_dsc_ise = self.SD_DSC_ISE_LS;
    
    '''
    symbol detection
    @Y:             the received signal in the delay Doppler domain [B, doppler, delay]
    @Xp:            the pilot in the delay Doppler domain [B, doppler, delay]
    @h:             initial channel estimation - path gains [B, Pmax]
    @hv:            initial channel estimation - variance [B, Pmax]
    @hm:            initial channel estimation - mask [B, Pmax]
    @No:            the noise (linear) power or a vector of noise [B] (variant noise power)
    @XdLocs(opt):   [B, doppler, delay]
    @sym_map(opt):  false by default. If true, the output will be mapped to the constellation
    '''
    def detect(self, Y, Xp, h, hv, hm, No, *, XdLocs=None, sym_map=False):
        if self.B is self.B0:
            raise Exception("The neural network requires batched data. Please set a batch size first.");
        # inputs
        Y = torch.as_tensor(Y).to(self.ctype).to(self.device)
        Xp = torch.as_tensor(Xp).to(self.ctype).to(self.device)
        # CSI
        h = torch.as_tensor(h).to(self.ctype).to(self.device)
        hm = torch.as_tensor(hm).to(self.device)
        hv = torch.as_tensor(hv).to(self.device)
        # noise
        No = torch.as_tensor(No).to(self.ftype).to(self.device)
        if No.ndim == 0:
            No = No.repeat(self.B)
        No = No[..., None, None]
            
        # optional inputs
        if XdLocs is None:  XdLocs = torch.ones([self.B, self.oc.K, self.oc.L], dtype=torch.bool)
        XdLocs = torch.as_tensor(XdLocs).to(self.device)
            
        # constant values
        y = torch.reshape(Y, [self.B, self.oc.sig_len, 1])
        xp = torch.reshape(Xp, [self.B, self.oc.sig_len, 1])
        xdlocs = XdLocs.reshape([self.B, self.oc.sig_len, 1])
        Hvm = xdlocs.movedim(-1, -2).repeat(1, self.oc.sig_len, 1)  # Hv mask (only count when symbol exist)
        # constant values (NN)
        if self.is_net:
            y_r = torch.cat([y.real, y.imag], -2)
            xp_r = torch.cat([xp.real, xp.imag], -2)
            xdlocs_r = xdlocs.repeat(1, 2, 1)
        
        # iterative detection - init
        ise_dsc_prev = torch.zeros([self.B, self.oc.sig_len, 1]).to(self.device)
        x_bse_prev = None
        v_bse_prev = None
        x_dsc = torch.zeros([self.B, self.oc.sig_len, 1], dtype=self.ctype).to(self.device)
        # iterative detection
        for iter_id in range(self.iter_num):
            # build the channel
            H, Hv = self.HtoDD(h, hv, hm)
            Ht = H.transpose(-1, -2).conj()
            Hty = Ht @ y
            HtH = Ht @ H
            HtH_off = self.off_diag*HtH
            sigma2_H = ((Hv*Hvm)*self.Ed).sum(-1, keepdim=True)     # CHE variance
            
            # Symbol Detection (SD)
            # SD - BSO
            # SD - BSO - mean
            if iter_id == 0:
                # SD - BSO - mean - 1st iter
                if self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_MMSE:
                    x_bso = torch.linalg.solve(
                        HtH + No/self.Ed*torch.eye(self.oc.sig_len).repeat(self.B, 1, 1), 
                        Hty - HtH_off @ x_dsc - HtH @ xp
                        )
                elif self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_MRC:
                    x_bso = 1/torch.norm(H, dim=-2).unsqueeze(-1)**2 * (Hty - HtH_off@x_dsc - HtH@xp);
                elif self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_LS:
                    x_bso = torch.linalg.solve(
                        HtH,
                        Hty - HtH_off @ x_dsc - HtH @ xp
                        )
            else:
                # SD - BSO - mean - other iteration
                if self.sd_bso_mean_cal == self.SD_BSO_MEAN_CAL_MRC:
                    x_bso = 1/torch.norm(H, dim=-2).unsqueeze(-1)**2 * (Hty - HtH_off@x_dsc - HtH@xp);
                elif self.sd_bso_mean_cal == self.SD_BSO_MEAN_CAL_LS:
                    x_bso = torch.linalg.solve(
                        HtH,
                        Hty - HtH_off@x_dsc - HtH@xp
                        )
            # SD - BSO - variance
            if self.sd_bso_var == self.SD_BSO_VAR_TYPE_APPRO:
                v_bso = 1/torch.norm(H, dim=-2).unsqueeze(-1)**2 * (No + sigma2_H)
            
            # SD - BSO - data filter
            x_bso = x_bso * xdlocs  # zero no data part (x_bso[~xdlocs]=0 detach gradient)
            v_bso = v_bso.clamp(self.min_var)
            v_bso = v_bso * xdlocs

            # BSE
            x_bso_d = x_bso[xdlocs].reshape(self.B, self.oc.data_len, 1)
            v_bso_d = v_bso[xdlocs].reshape(self.B, self.oc.data_len, 1)
            # BSE - Estimate P(x|y) using Gaussian distribution
            pxyPdfExpPower = -1/(2*v_bso_d)*abs(x_bso_d - self.constel_B_row)**2;
            # BSE - make every row the max power is 0
            #     - max only consider the real part
            pxypdfExpNormPower = pxyPdfExpPower - pxyPdfExpPower.max(-1, keepdim=True).values
            pxyPdf = torch.exp(pxypdfExpNormPower);
            # BSE - Calculate the coefficient of every possible x to make the sum of all
            pxyPdfCoeff = 1/pxyPdf.sum(-1, keepdim=True)
            # BSE - PDF normalisation
            pxyPdfNorm = pxyPdfCoeff*pxyPdf;
            # BSE - calculate the mean and variance
            x_bse_d = (pxyPdfNorm*self.constel_B_row).sum(-1, keepdim=True)
            v_bse_d = (abs(x_bse_d - self.constel_B_row)**2*pxyPdfNorm).sum(-1, keepdim=True)
            v_bse_d = v_bse_d.clamp(self.min_var)
            # BSE - resize
            x_bse = self.x_bse0.masked_scatter(xdlocs, x_bse_d)
            v_bse = self.v_bse0.masked_scatter(xdlocs, v_bse_d)

            # SD - DSC
            if self.sd_dsc_ise == self.SD_DSC_ISE_MRC:
                dsc_w = 1/torch.norm(H, dim=-2).unsqueeze(-1)**2
            ise_dsc = abs(dsc_w * (Hty - HtH@(x_bso + xp)))**2
            ies_dsc_sum = (ise_dsc + ise_dsc_prev).clamp(self.min_var)
            # DSC - rho (if we use this rho, we will have a little difference)
            rho_dsc = ise_dsc_prev/ies_dsc_sum
            # DSC - mean
            if iter_id == 0:
                x_dsc = x_bse
                v_dsc = v_bse
            else:
                x_dsc = (1 - rho_dsc)*x_bse_prev + rho_dsc*x_bse
                v_dsc = (1 - rho_dsc)*v_bse_prev + rho_dsc*v_bse
                
            # update statistics
            # update statistics - BSE
            x_bse_prev = x_bse
            v_bse_prev = v_bse
            # update statistics - DSC - instantaneous square error
            ise_dsc_prev = ise_dsc;
            
            # GNN - SD
            x_gnn = x_dsc
            v_gnn = v_dsc
            
            
            # CE
            X = x_gnn.reshape(self.B, self.oc.K, self.oc.L)
            V = v_gnn.reshape(self.B, self.oc.K, self.oc.L)
            
            Phi, PhiV = self.XtoPhi(X + Xp, V)
            Phi_M = Phi * hm[:, None, :]
            PhiTPhi = (Phi.transpose(-1, -2).conj() @ Phi) * hm[:, None, :] + No * torch.eye(self.pmax).repeat(self.B, 1, 1)
            Phi_pinv = torch.linalg.solve(PhiTPhi, Phi_M.transpose(-1, -2).conj())  
            h = hm[:, None, :] * (Phi_pinv @ y)
            
            # GNN - CE
            h_gnn = h
            hv_gnn = None
            
        
        # soft symbol estimation
        # x_det[xdlocs] = self.symmapNoBat(x_dsc[xdlocs]);
        # x_det[xndlocs] = 0;
        
        # only keep data part
        x = x_det[xdlocs] if self.batch_size is self.BATCH_SIZE_NO else np.reshape(x_det[xdlocs], (self.batch_size, -1));
        return x, H;
    
    ###########################################################################
    # AI functions
    ###########################################################################
    
    
    '''
    CHE - Factor to Variable: generate features
    @y:         the received signal tensor, [(batch_size), M*N, (1)]
    @Phi:       the channel estimation matrix tensor, [(batch_size), M*N, 2*p]
    @Phi_conff: the channel estimation matrix confidence tensor
    @no_arr:    the noise tensor, [(batch_size), 1, 2*p]
    '''
    def aiCheGenF2VFeats(self, y, Phi, no_arr):
        ytphi = torch.movedim(y, -1, -2) @ phi; # [(batch_size), 1, p], each element is yT @ phi_i
        phiTphi_diag = torch.sum(phi * phi.conj(), dim=-2, keepdim=True); 
        node_che = torch.movedim(torch.cat([ytphi, -1*phiTphi_diag, no_arr], dim=-2), -1, -2);
        edge_che = [];
        for i in range(self.P*2):
            edge_che_is = [];
            for j in range(self.P*2):
                if i != j:
                    edge_che_i_j = -1*(Phi[:, :, i].unsqueeze(-1).movedim(-1, -2)@Phi[:, :, j]);
                    edge_che_is.append(edge_che_i_j);
            edge_che.append(torch.cat(edge_che_is, dim=-1));
        edge_che = torch.cat(edge_che, dim=-2);
    ###########################################################################
    
    ###########################################################################
    # Auxiliary Methods
    ###########################################################################
    '''
    build Xp loc map
    '''
    def buildXpMap(self):
        # input check
        if self.Xp is None:
            raise Exception("The pilot matrix is not set.");
        elif self.Xp.ndim < 2 or self.Xp.ndim > 3:
            raise Exception("The pilot matrix dimension size must be 2 or 3.");
        Xp = self.Xp if self.Xp.ndim == 2 else self.Xp[0, ...];
        #
        self.XpMap = np.zeros([self.N, self.M], dtype=bool);
        for k in range(self.N):
            for l in range(self.M):
                self.XpMap[k, l] = True if abs(Xp[k, l]) > 1e-5 else False;
    
    '''
    build the channel estimation matrix
    @X:     the symbol estimation in DD domain      (B, doppler, delay)
    @V:     the esetimation variance in DD domain   (B, doppler, delay)
    '''
    def XtoPhi(self, X, V):
        Phi = []
        PhiV = []
        for yk in range(self.oc.K):
            for yl in range(self.oc.L):
                #Phi_ri = yk*self.oc.L + yl;      # row id in Phi
                Phi_r = []
                PhiV_r = []
                for p_id in range(self.pmax):
                    # path delay and doppler
                    li = self.lis[p_id].item()
                    ki = self.kis[p_id].item()
                    # x(k, l)
                    xl = yl - li
                    if yl < li:
                        xl = xl + self.oc.L;
                    xk = (yk - ki) % self.oc.K
                    # exponential part (pss_beta)
                    if self.oc.isPulIdeal():
                        pss_beta = torch.exp(-2j*torch.pi*li*ki/self.oc.L/self.oc.K)
                    elif self.oc.isPulRecta():
                        # here, you must use `yl-li` instead of `xl` or there will be an error
                        pss_beta = torch.exp(torch.as_tensor(2j*torch.pi*(yl - li)*ki/self.oc.L/self.oc.K))
                        if yl < li:
                            pss_beta = pss_beta*torch.exp(torch.as_tensor(-2j*torch.pi*xk/self.oc.K))
                    # assign value
                    Phi_r.append(X[..., xk, xl][..., None, None]*pss_beta)
                    PhiV_r.append(V[..., xk, xl][..., None, None])
                Phi_r = torch.cat(Phi_r, -1)
                PhiV_r = torch.cat(PhiV_r, -1)
                Phi.append(Phi_r)
                PhiV.append(PhiV_r)
        return torch.cat(Phi, -2), torch.cat(PhiV, -2)
    
    '''
    build Phi conffidence - the channel estimation matrix
    @Xp:    the Tx matrix in DD domain ([batch_size], doppler, delay)
    @lmax:  the maximal delay
    @kmax:  the maximal Doppler
    '''
    def buildPhiConf(self, Xp, Xd_var, lmax, kmax):
        
        pass
    
    '''
    H to DD domain [B, MN, MN]
    @h:         channel estimation - path gains [B, Pmax]
    @hv:        channel estimation - variance [B, Pmax]
    @hm:        channel estimation - mask [B, Pmax]
    '''
    def HtoDD(self, h, hv, hm):
        # init 
        H = self.H0
        Hv = self.Hv0
        # bi-orthogonal pulse
        if self.oc.isPulIdeal():
            return self.buildOtfsBiortDDChannel(h, hm)
        elif self.oc.isPulRecta():
            # rectangular pulse
            # accumulate all paths
            for tap_id in range(self.pmax):
                hmi = hm[..., tap_id]
                # only accumulate when there are at least a path
                if torch.any(hmi):
                    hi = h[..., tap_id]
                    hvi = hv[..., tap_id]
                    li = self.lis[tap_id].item()
                    ki = self.kis[tap_id].item() #np.expand_dims(kis[..., tap_id], axis=-1);
                    # delay
                    piMati = torch.roll(self.piMat, li, 1)
                    # Doppler            
                    timeSeq = torch.arange(-li, self.oc.sig_len-li).roll(li).repeat(self.B, 1).to(self.device)
                    deltaMat_diag = torch.exp( 2j*torch.pi*ki/(self.oc.sig_len)*timeSeq )
                    deltaMati = torch.diag_embed(deltaMat_diag)
                    # Pi, Qi & Ti
                    Pi = torch.einsum('...ij,...kl->...ikjl', self.dftmat, self.eyeL).reshape(self.B, self.oc.sig_len, self.oc.sig_len) @ piMati
                    Qi = deltaMati @ torch.einsum('...ij,...kl->...ikjl', self.idftmat, self.eyeL).reshape(self.B, self.oc.sig_len, self.oc.sig_len)
                    Ti = Pi @ Qi
                    # add this path
                    H = H + hi.reshape(-1, 1, 1) * Ti
                    Hv = Hv + hvi.reshape(-1, 1, 1) * abs(Ti)
            # to real
            #Hr = Hdd.real    # [B, m, n]
            #Hi = Hdd.imag    # [B, m, n]
            #top = torch.cat([Hr, -Hi], -1)      # [B, m, 2n]
            #bottom = torch.cat([Hi, Hr], -1)    # [B, m, 2n]
            #Hdd_r = torch.cat([top, bottom], -2)  # [B, 2m, 2n]
        # set the minimal variance
        Hv = Hv.clamp(self.min_var)
        return H, Hv
        
    ###########################################################################
    
    ###########################################################################
    # private methods
    ###########################################################################
    '''
    build the ideal pulse DD channel (callable after modulate)
    @h:     path gains [B, Pmax]
    @hm:    mask [B, Pmax]
    '''
    def buildOtfsBiortDDChannel(self, h, hm):
        # input check
        if self.pulse_type != self.PUL_BIORT:
            raise Exception("Cannot build the ideal pulse DD channel while not using ideal pulse.");
        hw = self.zeros(self.N, self.M).astype(complex);
        H_DD = self.zeros(self.sig_len, self.sig_len).astype(complex);
        for l in range(self.M):
            for k in range(self.N):
                    for tap_id in range(p):
                        if self.batch_size == self.BATCH_SIZE_NO:
                            hi = his[tap_id];
                            li = lis[tap_id];
                            ki = kis[tap_id];
                        else:
                            hi = np.expand_dims(his[..., tap_id], axis=-1);
                            li = np.expand_dims(lis[..., tap_id], axis=-1);
                            ki = np.expand_dims(kis[..., tap_id], axis=-1);
                        hw_add = 1/self.sig_len*hi*np.exp(-2j*np.pi*li*ki/self.sig_len)* \
                                np.expand_dims(np.sum(np.exp(2j*np.pi*(l-li)*self.seq(self.M)/self.M), axis=-1), axis=-1)* \
                                np.expand_dims(np.sum(np.exp(-2j*np.pi*(k-ki)*self.seq(self.N)/self.N), axis=-1), axis=-1);
                        if self.batch_size == self.BATCH_SIZE_NO:
                            hw[k, l]= hw[k, l] + hw_add;
                        else:
                            hw[..., k, l]= hw[...,k, l] + self.squeeze(hw_add);
                    if self.batch_size == self.BATCH_SIZE_NO:
                        H_DD = H_DD + hw[k, l]*self.kron(self.circshift(self.eye(self.N), k), self.circshift(self.eye(self.M), l));
                    else:
                        H_DD = H_DD + np.expand_dims(hw[..., k, l], axis=(-1,-2))*self.kron(self.circshift(self.eye(self.N), k), self.circshift(self.eye(self.M), l));
        return H_DD;
    
    '''
    build the rectangular pulse DD channel (callable after modulate)
    @h:     path gains [B, Pmax]
    @hm:    mask [B, Pmax]
    '''
    def HtoDD_Recta(self, h, hv, hm):
        
        return Hdd, Hddv
    
    
    '''
    symbol mapping (hard)
    '''
    def symmap(self, syms):
        syms = np.asarray(syms);
        if not self.isvector(syms):
            raise Exception("Symbols must be into a vector form to map.");
        syms_len = syms.shape[-1];
        syms_mat = self.repmat1(np.expand_dims(syms, -1), 1, self.constel_len);
        constel_mat = self.repmat1(self.constel, syms_len, 1);
        syms_dis = abs(syms_mat - constel_mat)**2;
        syms_dis_min_idx = syms_dis.argmin(axis=-1);
        
        return np.take(self.constel, syms_dis_min_idx);
    
    '''
    symbol mapping (hard, no batch)
    '''
    def symmapNoBat(self, syms):
        syms_len = syms.shape[-1];
        syms_mat = np.tile(np.expand_dims(syms, -1), (1, self.constel_len));
        constel_mat = self.repmat1(self.constel, syms_len, 1);
        syms_dis = abs(syms_mat - constel_mat)**2;
        syms_dis_min_idx = syms_dis.argmin(axis=-1);
        return np.take(self.constel, syms_dis_min_idx);
    ###########################################################################