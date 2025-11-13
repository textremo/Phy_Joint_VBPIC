class OTFSConfig(object):
    ###########################################################################
    # Constants
    # frame type
    FRAME_TYPE_GIVEN = -1;      # user input the frame matrix directly
    FRAME_TYPE_FULL = 0;        # full data
    FRAME_TYPE_CP = 1;          # using cyclic prefix between each two adjacent OTFS subframe (no interference between two adjacent OTFS subframes)
    FRAME_TYPE_ZP = 2;          # using zero padding (no interference between two adjacent OTFS subframes)
    FRAME_TYPES = [FRAME_TYPE_GIVEN, FRAME_TYPE_FULL, FRAME_TYPE_CP, FRAME_TYPE_ZP];
    # pulse type
    PUL_TYPE_IDEAL = 10;       # using ideal pulse to estimate the channel
    PUL_TYPE_RECTA = 20;       # using rectangular waveform to estimate the channel
    PUL_TYPES = [PUL_TYPE_IDEAL, PUL_TYPE_RECTA]; 
    # pilot types
    PIL_TYPE_GIVEN = -1;                        # user input the pilot matrix directly
    PIL_TYPE_NO = 0;                            # no pilots
    PIL_TYPE_EM_MID = 10;                       # embedded pilots - middle
    PIL_TYPE_EM_ZP = 11;                        # embedded pilots - zero padding areas
    PIL_TYPE_SP_MID = 20;                       # superimposed pilots - middle
    PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY = 21;    # superimposed pilots - multiple orthogonal along delay axis
    PIL_TYPES = [PIL_TYPE_GIVEN, PIL_TYPE_NO, PIL_TYPE_EM_MID, PIL_TYPE_EM_ZP, PIL_TYPE_SP_MID, PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY];
    
    ###########################################################################
    # Variables
    frame_type = FRAME_TYPE_FULL
    pul_type = PUL_TYPE_RECTA
    pil_type = PIL_TYPE_NO
    K = 0               # timeslot number
    L = 0               # subcarrier number
    # zero padding length (along delay axis at the end)
    zp_len = 0
    # pilot
    pk_len = 1;         # length on Dopller axis
    pl_len = 1;         # length on delay axis
    pk_num = 1;         # pilot area (CHE) num on Doppler axis
    pl_num = 1;         # pilot area (CHE) num on delay axis
    # guard lengths on delay Doppler axes (only available for )
    gl_len_neg = 0
    gl_len_pos = 0
    gk_len_neg = 0
    gk_len_pos = 0
    # energy
    Es_d = 0            # data energy
    Es_p = 0            # piloter energy
    # vectorized length
    sig_len = 0
    data_len = 0
    
    '''
    set the frame
    @frame_type:    frame type
    @K:             timeslote number
    @L:             subcarrier number
    @zp_len:        zero padding length
    '''
    def setFrame(self, frame_type, K, L, *, zp_len=0):
        # load inputs
        self.frame_type = frame_type;
        self.L = L;
        self.K = K;
        self.sig_len = L*K
        self.data_len = L*K;
        self.zp_len = zp_len;
        # input check
        if self.frame_type not in self.FRAME_TYPES:
            raise Exception("`frame_type` must be a type of `OTFSConfig.FRAME_TYPES`.");
        elif self.frame_type == self.FRAME_TYPE_ZP and (self.zp_len <= 0 or self.zp_len >= self.L):
            raise Exception("In zeor padding OTFS, `zp_len` must be positive and less than subcarrier number.");
        if self.L <= 0:
            raise Exception("Subcarrier number must be positive.");
        if self.K <= 0:
            raise Exception("Timeslot number must be positive.");
        
    '''
    set the pulse tye
    @pul_type: pulse type
    '''
    def setPul(self, pul_type):
        if pul_type not in self.PUL_TYPES:
            raise Exception("`pul_type` must be a type of `OTFSConfig.PUL_TYPES`");
        self.pul_type = pul_type;
    '''
    check the pulse type
    '''
    def isPulIdeal(self):
        return self.pul_type == self.PUL_TYPE_IDEAL;
    def isPulRecta(self):
        return self.pul_type == self.PUL_TYPE_RECTA;
        
    '''
    set the pilot
    @pil_type: pilote type
    @pk_len: the pilot length along the 
    '''
    def setPil(self, pil_type, *, pk_len=1, pl_len=1, pk_num=1, pl_num=1):
        # load inputs
        self.pil_type = pil_type
        self.pk_len = pk_len
        self.pl_len = pl_len
        self.pk_num = pk_num
        self.pl_num = pl_num
        # input check
        if self.frame_type == self.FRAME_TYPE_FULL and (self.pil_type == self.PIL_TYPE_EM_MID or self.pil_type == self.PIL_TYPE_EM_ZP):
            raise Exception("The embedded pilots are not allowed to use while the frame is set to full data.");
        if self.frame_type != self.FRAME_TYPE_CP and self.pil_type == self.PIL_TYPE_EM_ZP:
            raise Exception("The zero pading pilots are not allowed to use while the frame is not set to zero padding.");
        # check optional inputs based on pilot types
        if self.pil_type in [self.PIL_TYPE_EM_MID, self.PIL_TYPE_EM_ZP, self.PIL_TYPE_SP_MID, self.PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY]:
            if self.pk_len <= 0 or self.pk_len >= self.K:
                raise Exception("`pk_len` must be positive and less than the timeslot number.");
            if self.pl_len <= 0 or self.pl_len >= self.L:
                    raise Exception("`pl_len` must be positive and less than the subcarrier number.");
        if self.pil_type ==  OTFSConfig.PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY:
            if self.pk_num * self.pk_len <= 0 or self.pk_num * self.pk_len >= self.K:
                raise Exception("`pk_num` must be positive and short enough to put all pilots along the Doppler axis.");
            if self.pl_num * self.pl_len <= 0 or self.pl_num * self.pk_len >= self.L:
                raise Exception("`pk_num` must be positive and short enough to put all pilots along the delay axis.");
            
    '''
    set the guard (only available for embedded pilots)
    '''
    def setGuard(self, gl_len_neg, gl_len_pos, gk_len_neg, gk_len_pos):
        # load inputs
        self.gl_len_neg = gl_len_neg;
        self.gl_len_pos = gl_len_pos;
        self.gk_len_neg = gk_len_neg;
        self.gk_len_pos = gk_len_pos;
        # input check
        if self.pil_type != self.PIL_TYPE_EM_MID and self.pil_type != self.PIL_TYPE_EM_ZP:
            raise Exception("The guard is only available when using embedded pilots.");
        if self.pk_len <= 0 or self.pk_len >= self.K:
            raise Exception("`pk_len` must be positive and less than the timeslot number.");
        if self.pl_len <= 0 or self.pl_len >= self.L:
            raise Exception("`pl_len` must be positive and less than the subcarrier number.");
        