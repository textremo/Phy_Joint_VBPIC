classdef JPIC < dynamicprops
    % constants
    properties(Constant)
        % CE (channel estimation)
        CE_MRC = 1;             % maximum ratio combing
        CE_ZF = 2;              % zero forcing
        % SD (symbol detection)
        % SD - BSO
        % SD - BSO - mean
        % SD - BSO - mean - 1st iter
        SD_BSO_MEAN_CAL_INIT_MMSE   = 1;
        SD_BSO_MEAN_CAL_INIT_MRC    = 2;
        SD_BSO_MEAN_CAL_INIT_LS     = 3;
        % SD - BSO - mean - other iters
        SD_BSO_MEAN_CAL_MRC = 1;    
        SD_BSO_MEAN_CAL_LS  = 2;
        % SD - BSO - var
        SD_BSO_VAR_TYPE_APPRO = 1;
        SD_BSO_VAR_TYPE_ACCUR = 2;
        SD_BSO_VAR_CAL_MMSE   = 1;
        SD_BSO_VAR_CAL_MRC    = 2;
        SD_BSO_VAR_CAL_LS     = 3;
        % DSC
        % DSC - instantaneous square error
        DSC_ISE_MMSE    = 1;
        DSC_ISE_MRC     = 2;
        DSC_ISE_LS      = 3;
        % SD - OUT
        SD_OUT_BSE = 1;
        SD_OUT_DSC = 2;
    end
    % properties
    properties
        constel {mustBeNumeric}
        constel_len {mustBeNumeric}
        Ed = 1;                                             % energy of data (constellation average power)
        Eh = 1;                                             % energy of the channel
        % CE
        ce_type                 = JPIC.CE_MRC;
        % SD
        sd_bso_mean_cal_init    = JPIC.SD_BSO_MEAN_CAL_INIT_MMSE;
        sd_bso_mean_cal         = JPIC.SD_BSO_MEAN_CAL_MRC;
        sd_bso_var              = JPIC.SD_BSO_VAR_TYPE_APPRO;
        sd_bso_var_cal          = JPIC.SD_BSO_VAR_CAL_MRC;
        dsc_ise                 = JPIC.DSC_ISE_MRC;
        sd_out                  = JPIC.SD_OUT_DSC;
        % OTFS configuration
        oc = NaN;
        Xp = NaN;                           % pilots values (a matrix)
        XpMap = NaN;                        % the pilot map
        % control
        min_var{mustBeNumeric}  = eps;      % the default minimal variance is 2.2204e-16
        iter_num                = 10;       % maximal iteration
        % control - JPIC
        es                      = false;    % early stop
        es_thres                = eps;      % early stop threshold (abs)
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % general methods
    methods
        %{
        constructor
        @oc:                OTFS configuration
        @constel:           the constellation, a vector
        @lmax:              the maximal delay index
        @kmin:              the minimal Doppler index
        @kmax:              the maximal Doppler index
        @Eh:                the energy of the channel
        @min_var(opt):      the minimal variance.
        @iter_num(opt):     the maximal iteration
        @es(opt):           early stop
        @es_thres(opt):     early stop threshold (abs)
        %}
        function self = JPIC(oc, constel, Eh, lmax, kmin, kmax, varargin)
            % inputs
            self.oc = oc;
            if ~isvector(constel)
                error("The constellation must be a vector.");
            else
                self.constel = reshape(constel, 1, []);            % constellation must be a row vector or an 1D vector
                self.constel_len = length(constel);
                self.Ed = sum(abs(constel).^2)/self.constel_len;   % constellation average power
            end
            self.Eh = Eh;
            % optional inputs
            inPar = inputParser;
            addParameter(inPar,"min_var",  self.min_var,  @isnumeric);
            addParameter(inPar,"iter_num", self.iter_num, @isnumeric);
            addParameter(inPar,"es",       self.es,       @islogical);
            addParameter(inPar,"es_thres", self.es_thres, @isnumeric);
            inPar.KeepUnmatched = true;     
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            self.min_var    = inPar.Results.min_var;
            self.iter_num   = inPar.Results.iter_num;
            self.es         = inPar.Results.es;
            self.es_thres   = inPar.Results.es_thres;

            % buffer
            % constellation
            addprop(self, 'constel_B_row'); self.constel_B_row = self.constel;
            % delay & Doppler
            addprop(self, 'pmax'); self.pmax = (lmax+1)*(2*kmax+1); 
            addprop(self, 'lis');  self.lis = kron(0:lmax, ones(1, kmax-kmin+1));
            addprop(self, 'kis');  self.kis = repmat(kmin:kmax, 1, lmax+1);
            % H0
            addprop(self, 'H0');  self.H0 = zeros(self.oc.sig_len, self.oc.sig_len);
            addprop(self, 'Hv0'); self.Hv0 = zeros(self.oc.sig_len, self.oc.sig_len);
            
            % off-diagonal
            addprop(self, 'off_diag'); self.off_diag =  eye(self.oc.sig_len)+1 - eye(self.oc.sig_len).*2;
            % eye
            addprop(self, 'eyeKL'); self.eyeKL = eye(self.oc.sig_len);
            addprop(self, 'eyeK'); self.eyeK = eye(self.oc.K);
            addprop(self, 'eyeL'); self.eyeL = eye(self.oc.L);
            addprop(self, 'eyePmax'); self.eyePmax = eye(self.pmax);
            % ideal pulse
            if self.oc.isPulIdeal()
                addprop(self, 'hw0');  self.hw0 = zeros(self.oc.K, self.oc.L);
                addprop(self, 'hvw0'); self.hvw0 = zeros(self.oc.K, self.oc.L);
            end
            % rectangular pulse
            if self.oc.isPulRecta()
                % DFT matrix  
                addprop(self, 'dftmat'); self.dftmat = dftmtx(self.oc.K)*sqrt(1/self.oc.K);
                % IDFT matrix         
                addprop(self, 'idftmat'); self.idftmat = conj(self.dftmat);       
                % permutation matrix (from the delay) -> pi        
                addprop(self, 'piMat'); self.piMat = eye(self.oc.sig_len);            
            end
            % Symbol Detection
            addprop(self, 'x_bse0'); self.x_bse0 = zeros(self.oc.sig_len, 1);
            addprop(self, 'v_bse0'); self.v_bse0 = zeros(self.oc.sig_len, 1);
        end
        
        %{
        settings - CE
        %}
        function setCE2MRC(self)
            self.ce_type = self.CE_MRC;
        end
        function setCE2ZF(self)
            self.ce_type = self.CE_ZF;
        end

        %{
        settings - SD
        %}
        % settings - SD - BSO - mean - cal (init)
        function setSdBsoMealCalInit2MMSE(self)
            self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MMSE;
        end
        function setSdBsoMealCalInit2MRC(self)
            self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MRC;
        end
        function setSdBsoMealCalInit2LS(self)
            self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_LS;
        end
        % settings - SD - BSO - mean - cal
        function setSdBsoMeanCal2MRC(self)
            self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_MRC;
        end
        function setSdBsoMeanCal2LS(self)
            self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_LS;
        end
        
        %{
        symbol detection
        @Y:             the received signal in the delay Doppler domain [B, doppler, delay]
        @Xp:            the pilot in the delay Doppler domain [B, doppler, delay]
        @h:             initial channel estimation - path gains [B, Pmax]
        @hv:            initial channel estimation - variance [B, Pmax]
        @hm:            initial channel estimation - mask [B, Pmax]
        @No:            the noise (linear) power
        @XdLocs(opt):   [B, doppler, delay]
        @sym_map(opt):  false by default. If true, the output will be mapped to the constellation
        <DEBUG>
        @his:           the real path gain for each path
        @x:             the real symbols for each path
        %}
        function [x, H] = detect(self, Y, Xp, h, hv, hm, No, varargin)
            % register optional inputs 
            inPar = inputParser;
            addParameter(inPar,"XdLocs",  true(self.oc.K, self.oc.L));
            addParameter(inPar,"sym_map", false, @(x) isscalar(x)&islogical(x));
            addParameter(inPar,"his",     NaN,   @(x) isscalar(x)&islogical(x));
            addParameter(inPar,"x",       NaN,   @(x) isscalar(x)&islogical(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            XdLocs  = inPar.Results.XdLocs;
            sym_map = inPar.Results.sym_map;
            
            % constant values
            y = reshape(Y.', [], 1);
            xp = reshape(Xp.', [], 1);
            xdlocs = reshape(XdLocs.', [], 1);
            Hvm = repmat(xdlocs.', self.oc.sig_len, 1);
            PhiVm = repmat(hm(:).', self.oc.sig_len, 1);
            % iterative detection - intial params
            dsc_ise_prev = zeros(self.oc.sig_len, 1);
            dsc_ise_prev2 = zeros(self.pmax, 1);
            x_bse_prev = NaN;
            v_bse_prev = NaN;
            h_prev = zeros(self.pmax, 1);
            hv_prev = zeros(self.pmax, 1);
            x_dsc = zeros(self.oc.sig_len, 1);
            h_dsc = h;
            hv_dsc = hv;
            % iterative detection
            for iter_id = 1:self.iter_num
                % build the channel
                [H, Hv] = self.HtoDD(h_dsc, hv_dsc, hm);
                sigma2_H = sum((Hv.*Hvm)*self.Ed, 2);
                % noise whitening
                L = sqrt(sigma2_H + No);            % noise whitening matrix
                yw = y./L;
                H = H./L;   
                Ht = H';
                Hty = Ht*yw;
                HtH = Ht*H;
                HtH_off = self.off_diag.*HtH;
               
                % Symbol Detection (SD)
                % SD - BSO
                % SD - BSO - mean
                if iter_id == 1
                    % SD - BSO - mean - 1st iter
                    if self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_MMSE
                        x_bso = (HtH + self.eyeKL)\(Hty - HtH_off * x_dsc - HtH * xp);
                    elseif self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_MRC
                        x_bso = (Hty - HtH_off * x_dsc - HtH * xp)./sum(abs(H).^2).'; 
                    elseif self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_LS
                        x_bso = HtH\(Hty - HtH_off * x_dsc - HtH * xp);
                    end
                else
                    % SD - BSO - mean - other iteration
                    if self.sd_bso_mean_cal == self.SD_BSO_MEAN_CAL_MRC
                        x_bso = (Hty - HtH_off * x_dsc - HtH * xp)./sum(abs(H).^2).';
                    elseif self.sd_bso_mean_cal == self.SD_BSO_MEAN_CAL_LS
                        x_bso = HtH\(Hty - HtH_off * x_dsc - HtH * xp);
                    end
                end
                % SD - BSO - variance
                if self.sd_bso_var == self.SD_BSO_VAR_TYPE_APPRO
                    v_bso = 1./sum(abs(H).^2).';
                end
                % SD - BSO - data filter
                x_bso = x_bso.*xdlocs;  % zero no data part (x_bso[~xdlocs]=0 detach gradient)
                v_bso = max(v_bso, self.min_var);
                v_bso = v_bso.*xdlocs;

                % SD - BSE
                x_bso_d = x_bso(xdlocs);
                v_bso_d = v_bso(xdlocs);
                % BSE - Estimate P(x|y) using Gaussian distribution
                pxyPdfExpPower = -1./(2*v_bso_d).*abs(x_bso_d - self.constel_B_row).^2;
                % BSE - make every row the max power is 0
                %     - max only consider the real part
                pxypdfExpNormPower = pxyPdfExpPower - max(pxyPdfExpPower, [], 2);
                pxyPdf = exp(pxypdfExpNormPower);
                % BSE - Calculate the coefficient of every possible x to make the sum of all
                pxyPdfCoeff = 1./sum(pxyPdf, 2);
                %pxyPdfCoeff = repmat(pxyPdfCoeff, 1, self.constel_len);
                % BSE - PDF normalisation
                pxyPdfNorm = pxyPdfCoeff.*pxyPdf;
                % BSE - calculate the mean and variance
                x_bse_d = sum(pxyPdfNorm.*self.constel_B_row, 2);
                v_bse_d = sum(abs(x_bse_d - self.constel_B_row).^2.*pxyPdfNorm, 2);
                v_bse_d = max(v_bse_d, self.min_var);
                % BSE - resize
                x_bse = self.x_bse0;
                v_bse = self.v_bse0;
                x_bse(xdlocs) = x_bse_d;
                v_bse(xdlocs) = v_bse_d;

                % DSC - SD
                if self.dsc_ise == self.DSC_ISE_MRC
                    dsc_w = 1./sum(abs(H).^2).';
                    ise_dsc = abs(dsc_w.*(Hty - HtH*(x_bso + xp))).^2;
                end
                ies_dsc_sum = max(ise_dsc + dsc_ise_prev, self.min_var);
                % DSC - rho (if we use this rho, we will have a little difference)
                rho_dsc = dsc_ise_prev./ies_dsc_sum;
                % DSC - mean
                if iter_id == 1
                    x_dsc = x_bse;
                    v_dsc = v_bse;
                else
                    x_dsc = (1 - rho_dsc).*x_bse_prev + rho_dsc.*x_bse;
                    v_dsc = (1 - rho_dsc).*v_bse_prev + rho_dsc.*x_bse;
                end

                % update statistics
                % update statistics - BSE
                x_bse_prev = x_bse;
                v_bse_prev = v_bse;
                % update statistics - DSC - instantaneous square error
                dsc_ise_prev = ise_dsc;

                % CE
                X = reshape(x_dsc, self.oc.L, self.oc.K).';
                V = reshape(v_dsc, self.oc.L, self.oc.K).';
                [Phi, PhiV] = self.XtoPhi(X + Xp, V);
                sigma2_Phi = sum((PhiV.*PhiVm)*self.Eh, 2);
                L = sqrt(sigma2_Phi + No);            % noise whitening matrix
                Phiwm = Phi./L.*hm.';
                yw = y./L;
                h = (Phiwm'*Phiwm + self.eyePmax) \ (Phiwm'*yw);
                hv = 1./sum(abs(Phiwm).^2).';

                % DSC - CE
                if self.dsc_ise == self.DSC_ISE_MRC
                    dsc_w = 1./sum(abs(Phi).^2).';
                    ise_dsc2 = abs(dsc_w.*(Phi'*y - Phi'*Phi*h)).^2;
                end
                ies_dsc_sum2 = max(ise_dsc2 + dsc_ise_prev2, self.min_var);
                rho_dsc2 = dsc_ise_prev2./ies_dsc_sum2;
                if iter_id == 1
                    h_dsc = h;
                    hv_dsc = hv;
                else
                    h_dsc = (1 - rho_dsc2).*h_prev + rho_dsc2.*h;
                    hv_dsc = (1 - rho_dsc2).*hv_prev + rho_dsc2.*hv;
                end
                % update statistics
                h_prev = h_dsc;
                hv_prev = hv_dsc;
                % update statistics - DSC - instantaneous square error
                dsc_ise_prev2 = ise_dsc2;
                % h_dsc = h;
                % hv_dsc = hv;

            end
            % only keep data part
            x = x_dsc(xdlocs);
            [H, ~] = self.HtoDD(h, hv, hm);
            % hard estimation
            if sym_map
                x = self.symmap(x);
            end
        end
    end


    %%---------------------------------------------------------------------
    % private methods
    methods(Access=private)
        %{
        H to DD domain [B, MN, MN]
        @h:         channel estimation - path gains [B, Pmax]
        @hv:        channel estimation - variance [B, Pmax]
        @hm:        channel estimation - mask [B, Pmax]
        %}
        function [H, Hv] = HtoDD(self, h, hv, hm)
            % init 
            H = self.H0;
            Hv = self.Hv0;
            % build the DD channel
            if self.oc.isPulIdeal()
                % ideal pulse
                hw = self.hw0;
                for l = 1:self.oc.L
                    for k = 1:self.oc.K
                        for tap_id = 1:p
                            hi = h(tap_id);
                            li = self.lis(tap_id);
                            ki = self.kis(tap_id);
                            hw_add = 1/self.oc.sig_len*hi*exp(-2j*pi*li*ki/self.oc.sig_len)* ...
                                    sum(exp(2j*pi*(l-li)*(0:self.oc.L-1)/self.oc.L))* ...
                                    sum(exp(-2j*pi*(k-ki)*(0:self.oc.K-1)/self.oc.K));
                            hw(k, l)= hw(k, l) + hw_add;
                        end
                        H = H + hw(k, l)*kron(circshift(self.eyeK, k), circshift(self.eyeL, l));
                    end
                end
            elseif self.oc.isPulRecta()
                % rectangular pulse
                % accumulate all paths
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
                        timeSeq = [0:self.oc.sig_len-1-li, -li:-1];
                        deltaMat_diag = exp(2j*pi*ki/(self.oc.sig_len)*timeSeq);
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
            Hv = max(Hv, self.min_var);
        end

        %{
        build the channel estimation matrix
        @X:     the symbol estimation in DD domain      (doppler, delay)
        @V:     the esetimation variance in DD domain   (doppler, delay)
        %}
        function [Phi, PhiV] = XtoPhi(self, X, V)
            Phi = zeros(self.oc.sig_len, self.pmax);
            PhiV = zeros(self.oc.sig_len, self.pmax);
            for yl = 1:self.oc.L
                for yk = 1:self.oc.K
                    Phi_ri = (yl - 1)*self.oc.K + yk;
                    for p_id = 1:self.pmax
                        li = self.lis(p_id);
                        ki = self.kis(p_id);
                        % x(k, l)
                        xl = yl - li;
                        if yl - 1 < li
                            xl = xl + self.oc.L;
                        end
                        xk = mod(yk - 1 - ki, self.oc.K) + 1;
                        % exponential part (pss_beta)
                        if self.oc.isPulIdeal()
                            pss_beta = exp(-2j*pi*li/self.oc.L*ki/self.oc.K);
                        elseif self.oc.isPulRecta()
                            % here, you must use `yl-li` instead of `xl` or there will be an error
                            pss_beta = exp(2j*pi*(yl-li-1)/self.oc.L*ki/self.oc.K);
                            if yl - 1 < li
                                pss_beta = pss_beta*exp(-2j*pi*(xk - 1)/self.oc.K);
                            end
                        end
                        % assign value
                        try
                            Phi(Phi_ri, p_id) = X(xk, xl)*pss_beta;
                            PhiV(Phi_ri, p_id) = V(xk, xl);
                        catch
                            disp();
                        end
                    end
                end
            end
        end

        %{
        symbol mapping
        %}
        function syms_mapped = symmap(self, syms)
            if ~isvector(syms)
                error("Symbols must be into a vector form to map.");
            end
            % the input must be a column vector
            is_syms_col = iscolumn(syms);
            syms = syms(:);
            syms_dis = abs(syms - self.constel).^2;
            [~,syms_dis_min_idx] =  min(syms_dis,[],2);
            syms_mapped = self.constel(syms_dis_min_idx);
            if is_syms_col
                syms_mapped = syms_mapped(:);
            end
        end
    end

    %%---------------------------------------------------------------------
    % debug
    methods(Access=public)
    end
end