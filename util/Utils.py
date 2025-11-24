import numpy as np

'''
generate the full list of H
@lmax:  the lmax in channel estimation (instead of real lmax)
@kmax:  the kmax in channel estimation (instead of real kmax)
'''
def realH2Hfull(kmax, lmax, his, lis, kis, *, batch_size=None):
    his_new = np.zeros((2*kmax+1)*(lmax+1), dtype=complex) if batch_size is None else np.zeros([batch_size, (2*kmax+1)*(lmax+1)], dtype=complex);
    p_len = len(his) if batch_size is None else his.shape[-1];
    for p_id in range(p_len):
        hi = his[p_id] if batch_size is None else his[..., p_id];
        li = lis[p_id] if batch_size is None else lis[..., p_id];
        ki = kis[p_id] if batch_size is None else kis[..., p_id];
        #pos = (ki + kmax)*(lmax + 1) + li;
        pos = li*(2*kmax+1) + kmax + ki
        if pos.ndim < 1:
            his_new[pos] = hi;
        else:
            for b_id in range(batch_size):
                his_new[b_id, pos[b_id]] = hi[b_id];

            
    return his_new;


'''
get his full list as the format below
    path gains:     h0, h1, h2, ... 
    delay:          l0, l0, l0, ..., l0,   l1, l1, l1, l1, ..., l1,   ...
    Doppler:        k0, k1, k2, ..., kmax, k0, k0, k1, k2, ..., kmax, ...
@his:   the path gains
@lis:   the delay
@kis:   the doppler 
@lmax:  the maximal delay index
@kmax:  the maximal Doppler index
@B(opt): batch size
'''
def getHisFullList(his, lis, kis, lmax, kmax, *, B=None):
    his = np.asarray(his);
    lis = np.asarray(lis);
    kis = np.asarray(kis);
    # input check
    if B is None and his.ndim != 1 or B is not None and his.ndim != 2:
        raise Exception("The path gain must be a vector.")
    if B is None and lis.ndim != 1 or B is not None and lis.ndim != 2:
        raise Exception("The delay must be a vector.")
    if B is None and kis.ndim != 1 or B is not None and kis.ndim != 2:
        raise Exception("The Doppler must be a vector.")
    if his.shape[-1] != lis.shape[-1]:
        raise Exception("The delay should be the same dimension with the path gain.");
    if his.shape[-1] != kis.shape[-1]:
        raise Exception("The Doppler should be the same dimension with the path gain.");
    # generate the path gain list
    p_len = his.shape[-1];
    batch_size = 1 if B is None else B;
    pmax = (2*kmax+1)*(lmax+1);
    his_new = np.zeros([batch_size, pmax], dtype=complex);
    # generate the list 
    his_mask = np.zeros([batch_size, pmax], dtype=bool);
    # match input dimensions
    if B is None:
        his = his[np.newaxis, :]
        lis = lis[np.newaxis, :]
        kis = kis[np.newaxis, :]
    # fill the new path gain list
    for bid in range(batch_size):
        for pid in range(p_len):
            hi = his[bid, pid]
            li = lis[bid, pid]
            ki = kis[bid, pid]
            hi_shift = li*(2*kmax+1) + kmax + ki
            his_new[bid, hi_shift] = hi
            his_mask[bid, hi_shift] = True
    if B is None:
        his_new = his_new.squeeze(0)
        his_mask = his_mask.squeeze(0)
        
    return his_new, his_mask