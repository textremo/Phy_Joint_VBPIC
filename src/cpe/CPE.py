from scipy.stats.distributions import chi2
import numpy as np

eps = np.finfo(np.float64).eps

class CPE(object):
    # constants
    EST_TYPE_LS      = 0
    EST_TYPE_MMSE    = 1
    EST_TYPES       = [EST_TYPE_LS, EST_TYPE_MMSE]
    PIL_VAL = (1+1j)*np.sqrt(0.5)
    BATCH_SIZE_NO = None;
    
    # variables
    oc = None;
    lmax = 0
    kmin = 0
    kmax = 0
    area_num = 0;
    pk = 0
    pls = None
    Ep = 0
    Ed = 0
    No = 0
    batch_size = 0;
    # variables - pilots
    pil_val = None;
    area_len = 0;
    rho = 0;        # the probability that the signal is 
    thres = 0;      # the threshold that this point is not noise (in power)
    
    '''
    @oc: OTFS configuration
    @lmax: the maximal delay index
    @kmax: the maximal Doppler index
    @No: the noise power
    '''
    def __init__(self, oc, lmax, kmax, Ed, No, *, B=None):
        self.oc = oc;
        self.lmax = lmax
        self.area_len = lmax + 1;
        self.area_num = np.floor(oc.L / self.area_len).astype(int);
        if self.area_num <= 0:
            raise Exception("There is no space for pilots.");
        # calculate the Doppler positions
        self.pk = np.floor((self.oc.K - 1)/2).astype(int);
        # update the k (Doppler) range
        self.kmin = -kmax if self.pk - self.kmax > 0 else -self.pk
        self.kmax =  kmax if self.pk + kmax < self.oc.K else self.oc.K - self.pk
        # calculate the delay positions
        self.pls = np.arange(self.area_num)*(lmax + 1)
        # calculate the threshold
        self.Ed = Ed
        self.No = No
        self.rho = chi2.ppf(0.9999, 2*self.area_num)/(2*self.area_num)
        self.thres = self.rho*(Ed + No)
        self.batch_size = B
        
    '''
    get the Doppler range
    '''
    def getKRange(self):
        return self.kmin, self.kmax
        
    '''
    generate pilots
    @Ep: the pilot energy
    '''
    def genPilots(self, Ep):
        self.Ep = Ep;
        self.pil_val = self.PIL_VAL*np.sqrt(Ep);
        Xp = np.zeros([self.oc.K, self.oc.L], dtype=complex) if self.batch_size is self.BATCH_SIZE_NO else np.zeros([self.batch_size, self.oc.K, self.oc.L], dtype=complex)
        Xp[..., self.pk, self.pls] = self.pil_val
        return Xp;
    
    '''
    estimate the path (h, k, l)
    @Y_DD:              the received resrouce grid (N, M)
    @is_all(opt):       estimate paths on all locations
    @est_type(opt):     the estimation type
    '''    
    def estPaths(self, Y_DD, *, is_all=False, est_type=EST_TYPE_LS):
        # input check
        if est_type not in self.EST_TYPES:
            raise Exception("CHE type is not supported.");
        
        # we sum all area together
        est_area = np.zeros([self.oc.K, self.area_len], dtype=complex) if self.batch_size is self.BATCH_SIZE_NO else np.zeros([self.batch_size, self.oc.K, self.area_len], dtype=complex);
        ca_id_beg = 0;
        ca_id_end = self.area_len;
        # accumulate all areas together
        for area_id in range(self.area_num):
            # build the phase matrix
            est_area = est_area + Y_DD[..., :, ca_id_beg:ca_id_end]*self.getPhaseConjMat(area_id);
            ca_id_beg = ca_id_end;
            ca_id_end = ca_id_end + self.area_len;
        est_area = est_area/self.area_num;
        # locate the searching space for ki
        ki_beg = self.pk + self.kmin 
        ki_end = self.pk + self.kmax + 1
        # find paths
        his = []
        his_var = []    # estimation error
        his_mask = []
        kis = []
        lis = []
        for l_id in range(0, self.area_len):
            for k_id in range(ki_beg, ki_end):
                pss_ys = np.expand_dims(est_area[k_id, l_id], axis=0) if self.batch_size == self.BATCH_SIZE_NO else est_area[..., k_id, l_id];
                pss_ys_ids_yes = abs(pss_ys)**2 > self.thres;
                pss_ys_ids_not = abs(pss_ys)**2 <= self.thres;
                li = l_id - self.pls[0];
                ki = k_id - self.pk;
                # estimate the channel
                if est_type == self.EST_TYPE_LS:
                    hi = pss_ys/self.pil_val;
                    hi_var = np.tile(self.Ed/self.Ep + self.No/self.Ep, pss_ys.shape)
                # zero values under the threshold
                hi[pss_ys_ids_not] = 0
                hi_var[pss_ys_ids_not] = eps
                hi_mask = pss_ys_ids_yes;
                # at least we find one path
                if is_all or np.sum(pss_ys_ids_yes, axis=None) > 0:
                    if self.batch_size != self.BATCH_SIZE_NO:
                        hi = hi[..., np.newaxis]
                        hi_var = hi_var[..., np.newaxis]
                        hi_mask = hi_mask[..., np.newaxis]
                        li = np.tile(li, (self.batch_size, 1));
                        ki = np.tile(ki, (self.batch_size, 1));
                    his.append(hi)     
                    his_var.append(hi_var)
                    his_mask.append(hi_mask)
                    lis.append(li)
                    kis.append(ki)
        his = np.concatenate(his, -1)
        his_var = np.concatenate(his_var, -1)
                    
        # return
        if is_all:
            his_mask = np.concatenate(his_mask, -1)
            return his, his_var, his_mask
        else:
            lis = np.asarray(lis) if self.batch_size == self.BATCH_SIZE_NO else np.concatenate(lis, -1)
            kis = np.asarray(kis) if self.batch_size == self.BATCH_SIZE_NO else np.concatenate(kis, -1)
            return his, his_var, lis, kis;
        
    ###########################################################################
    # private methods
    '''
    build the phase Conj matrix
    @area_id: the area index
    '''
    def getPhaseConjMat(self, area_id):
        if area_id < 0 or area_id >= self.area_num:
            raise Exception("Area Id is illegal.");
        phaseConjMat = np.zeros((self.oc.K, self.area_len), dtype=complex);
        for ki in range(0, self.oc.K):
            for li in range(0, self.area_len):
                if self.oc.isPulIdeal():
                    phaseConjMat[ki, li] = np.exp(2j*np.pi*li*(ki-self.pk)/self.oc.L/self.oc.K);
                elif self.oc.isPulRecta():
                    phaseConjMat[ki, li] = np.exp(-2j*np.pi*self.pls[area_id]*(ki-self.pk)/self.oc.L/self.oc.K);
                else:
                    raise Exception("The pulse type is unkownn.");
        if self.batch_size is not self.BATCH_SIZE_NO:
            phaseConjMat = np.tile(phaseConjMat, (self.batch_size, 1, 1));
        
        return phaseConjMat;