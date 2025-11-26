classdef Modu < dynamicprops
    properties(Constant)
        % OFDM 0~49
        MODU_OFDM_STD               = 0;
        MODUS_OFDM = [Modu.MODU_OFDM_STD];
        % OTFS 50~99
        MODU_OTFS_FULL          = 50;           % full
        MODU_OTFS_EMBED         = 51;           % embed
        MODU_OTFS_SP            = 60;           % superimposed
        MODU_OTFS_SP_REP_DELAY  = 65;           % superimposed - replicate on the delay axis
        MODUS_OTFS = [Modu.MODU_OTFS_FULL, Modu.MODU_OTFS_EMBED, Modu.MODU_OTFS_SP, Modu.MODU_OTFS_SP_REP_DELAY];
        % all
        MODUs = [Modu.MODUS_OFDM, ... 
                 Modu.MODUS_OTFS];

        % Frame type
        FT_CP = 1;              % cyclic prefix
        FT_ZP = 2;              % zero padding
        FTs = [Modu.FT_CP, Modu.FT_ZP];

        % Pulse Type
        PUL_BIORT = 0;          % biorthogonal
        PUL_RECTA = 1;          % rectangular pulse
        PULs = [Modu.PUL_BIORT, ... 
                Modu.PUL_RECTA];
    end
    properties
        %------------------------------------------------------------------
        % modulation
        modu = NaN;
        frame = NaN;
        pul = NaN;
        N = 0;              % timeslot number
        M = 0;              % subcarrier number
        % modulation - OTFS
        K = 0;              % Doppler (timeslot) number
        L = 0;              % delay (subcarrier) number
        % pilot
        dataLocs = NaN;     % data locations
        refSig = NaN;       % reference siganl (pilot + guard)
        csiLim = NaN;
        % vectorized length
        sig_len = 0;
        data_len = 0;
        
        %------------------------------------------------------------------
        % CSI
        Eh = NaN;                                           % energy of the channel

        %------------------------------------------------------------------
        % OTFS
        % CSI
        pmax = NaN;
        lis = NaN;
        kis = NaN;
        % area division
        pilCheRng = NaN;       % pilot CHE range [k0, kN, l0, lN] (k0->kN: k range, l0-lN: l range)            
        pilCheRng_klen = 0;
        pilCheRng_len = 0;
        % others
        H0 = NaN;
        Hv0 = NaN;
        off_diag = NaN;
        eyeKL = NaN;
        eyeK = NaN;
        eyeL = NaN;
        eyePmax = NaN;
        hw0 = NaN;
        hvw0 = NaN;
        dftmat = NaN;
        idftmat = NaN;
        piMat = NaN;
    end


    methods
        %{
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
        %}
        function self = Modu(modu, frame, pul, nTimeslot, nSubcarr, varargin)
            if ~ismember(modu, self.MODUs)
                error("The modulation type is not supported!!!");
            end
            if ~ismember(frame, self.FTs)
                error("The frame type is not supported!!!");
            end
            if ~ismember(pul, self.PULs)
                error("The pulse type is not supported!!!");
            end
            self.modu = modu;
            self.frame = frame;
            self.pul = pul;
            % dimension
            if modu < 50
                % OFDM
                self.N = nTimeslot;
                self.M = nSubcarr;
            else
                % OTFS
                self.K = nTimeslot;
                self.L = nSubcarr;
            end
            self.sig_len = nTimeslot*nSubcarr;
            if length(varargin) >= 1
                self.csiLim = varargin{1};
            end
            self.init();
        end

        %{
        init
        %}
        function init(self)
            if ~isnan(self.csiLim)
                lmax = self.csiLim(1); kmax = self.csiLim(2);
                % delay & Doppler
                self.pmax = (lmax+1)*(2*kmax+1); 
                self.lis = kron(0:lmax, ones(1, 2*kmax+1));
                self.kis = repmat(-kmax:kmax, 1, lmax+1);
                % H0
                self.H0 = zeros(self.sig_len, self.sig_len);
                self.Hv0 = zeros(self.sig_len, self.sig_len);
                % off-diagonal
                self.off_diag =  eye(self.sig_len)+1 - eye(self.sig_len).*2;
                % eye
                self.eyeKL = eye(self.sig_len);
                self.eyeK = eye(self.K);
                self.eyeL = eye(self.L);
                self.eyePmax = eye(self.pmax);

                % others
                switch self.pul
                    % bi-orthogonal pulse
                    case self.PUL_BIORT
                        self.hw0 = zeros(self.K, self.L);
                        self.hvw0 = zeros(self.K, self.L);
                    % rectangular pulse
                    case self.PUL_RECTA
                        self.dftmat = dftmtx(self.K)*sqrt(1/self.K);    % DFT matrix  
                        self.idftmat = conj(self.dftmat);               % IDFT matrix     
                        self.piMat = eye(self.sig_len);                 % permutation matrix (from the delay) -> pi
                end
            end
        end
    
        %{
        set the data location
        @dataLocs:       data locations, a 01 matrix of [N, M] or [K, L]
        %}
        function setDataLoc(self, dataLocs)
            self.dataLocs = dataLocs;
            self.data_len = sum(dataLocs, "all");
        end

        %{
        set the reference signal
        @refSig:         the reference sigal of [N, M] or [K, L], 0 at non-ref locations
        %}
        function setRef(self, refSig)
            self.refSig = refSig;
            if self.modu == self.MODU_OTFS_FULL
                error("Full data does not support any reference signal!!!");
            end
            
            % pilot CHE range
            if ~isnan(self.csiLim)
                lmax = self.csiLim(1); kmax = self.csiLim(2);
                refSig = self.refSig;
                if self.modu == self.MODU_OTFS_SP_REP_DELAY
                    refSig = self.refSig(:, 1:lmax+1);
                end
                [pk0, pl0] = find(abs(refSig) > eps, 1);
                [pkN, plN] = find(abs(refSig) > eps, 1, "last");
                self.pilCheRng = [max(pk0-kmax, 1), min(pkN+kmax, self.K), pl0, min(plN + lmax, self.L)];
                self.pilCheRng_klen = self.pilCheRng(2) - self.pilCheRng(1) + 1;
                self.pilCheRng_len = self.pilCheRng_klen*(self.pilCheRng(4) - self.pilCheRng(3) + 1);
            end
        end

        %{
        set the csi if know
        @Eh:                the energy of each path
        %}
        function setCSI(self, Eh)
            self.Eh = Eh;
        end

        %{
        check the pulse type
        %}
        function chk = isPulBiort(self)
            chk = self.pul == self.PUL_BIORT;
        end
        function chk = isPulRecta(self)
            chk = self.pul == self.PUL_RECTA;
        end


    end

    %----------------------------------------------------------------------
    % OTFS
    methods
        %{
        h to H (time domain to DD domain)
        @h:        CHE path gains
        <OPT>
        @hv:       CHE variance
        @hm:       CHE mask
        @min_var:  the minimal variance
        %}
        function [H, Hv] = h2H(self, h, varargin)
            % inputs
            inPar = inputParser;
            addParameter(inPar, "hv", []);
            addParameter(inPar, "hm", []);
            addParameter(inPar, "min_var",  eps,  @isnumeric);
            inPar.KeepUnmatched = true;     
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            hv = inPar.Results.hv;
            hm = inPar.Results.hm;
            min_var = inPar.Results.min_var;
            if isempty(hv)
                hv = ones(self.pmax, 1);
            end
            if isempty(hm)
                hm = ones(self.pmax, 1);
            end
            
            % to H
            H = self.H0;
            Hv = self.Hv0;
            switch(self.pul)
                case self.PUL_BIORT
                case self.PUL_RECTA
                    for tap_id = 1:self.pmax
                        hmi = hm(tap_id);
                        % only accumulate when there are at least a path
                        if hmi
                            hi = h(tap_id);
                            hvi = hv(tap_id);
                            li = self.lis(tap_id);
                            ki = self.kis(tap_id);
                            % delay
                            piMati = circshift(self.piMat, li); 
                            % Doppler
                            timeSeq = circshift(-li:self.sig_len-1-li, -li);
                            deltaMat_diag = exp(2j*pi*ki/(self.sig_len)*timeSeq);
                            deltaMati = diag(deltaMat_diag);
                            % Pi, Qi, & Ti
                            Pi = kron(self.dftmat, self.eyeL)*piMati; 
                            Qi = deltaMati*kron(self.idftmat, self.eyeL);
                            Ti = Pi*Qi;
                            H = H + hi*Ti;
                            Hv = Hv + hvi*abs(Ti);
                        end
                    end
            end
            % set the minimal variance
            Hv = max(Hv, min_var);
        end

        %{
        refSig to Phi
        %}
        function Phi = ref2Phi(self)
            if self.modu == self.MODU_OTFS_FULL
                error("Not refence signal is given on the full data frame type!!!");
            end
            
            Phi = zeros(self.pilCheRng_len, self.pmax);
            for yk = self.pilCheRng(1):self.pilCheRng(2)
                for yl = self.pilCheRng(3):self.pilCheRng(4)
                    Phi_ri = (yl - self.pilCheRng(3))*self.pilCheRng_klen + (yk - self.pilCheRng(1) + 1);
                    for p_id = 1:self.pmax
                        li = self.lis(p_id);
                        ki = self.kis(p_id);
                        % x(k, l)
                        xl = yl - li;
                        xk = yk - ki;
                        if abs(self.refSig(xk, xl)) > eps
                            % exponential part (pss_beta)
                            if self.isPulBiort()
                                pss_beta = exp(-2j*pi*li/self.L*ki/self.K);
                            elseif self.isPulRecta()
                                pss_beta = exp(2j*pi*(yl-li-1)/self.L*ki/self.K);     % here, you must use `yl-li` instead of `xl` or there will be an error
                            end
                            Phi(Phi_ri, p_id) = self.refSig(xk, xl)*pss_beta;
                        end
                    end
                end
            end
        end

        %{
        X to Phi
        %}
        function Phi = X2Phi(self)
        end
    end
end