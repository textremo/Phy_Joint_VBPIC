classdef ModuOTFS < Modu
    properties
        % CSI
        pmax = NaN;
        lis = NaN;
        kis = NaN;
        % area division
        pilCheRng = NaN;       % pilot CHE range [k0, kN, l0, lN] (k0->kN: k range, l0-lN: l range)            
        
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
        function self = ModuOTFS(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin)
            self = self@Modu(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin{:});
            if ~isnan(self.csiLim)
                lmax = self.csiLim(1); kmax = self.csiLim(2);
                % delay & Doppler
                self.pmax = (lmax+1)*(2*kmax+1); 
                self.lis = kron(0:lmax, ones(1, 2*kmax+1));
                self.kis = repmat(-kmax:kmax, 1, lmax+1);
                % pilot CHE range
                if self.modu ~= self.MODU_OTFS_FULL
                    refSig = self.refSig;
                    if self.modu == MODU_OTFS_SP_REP_DELAY
                        refSig = self.refSig(:, 1:lmax+1);
                    end
                    [pk0, pl0] = find(abs(refSig) > eps, 1);
                    [pkN, plN] = find(abs(refSig) > eps, 1, "last");
                    self.pilCheRng = [max(pk0-kmax, 1), min(pkN+kmax, self.K), pl0, min(plN + lmax, self.L)];
                end
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
                            timeSeq = [0:self.sig_len-1-li, -li:-1];
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
        function Phi = ref2Phi()
            
        end
    end
end