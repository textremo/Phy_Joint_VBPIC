% channel parameter estimation (CPE)
classdef CPE < handle
    % constants
    properties(Constant)
        EST_TYPE_LS     = 0
        EST_TYPE_MMSE   = 1
        EST_TYPES       = [CPE.EST_TYPE_LS, CPE.EST_TYPE_MMSE]
        PIL_VAL         = (1+1j)*sqrt(0.5)
    end
    % properties
    properties
        oc = NaN;       % OTFS configuration
        lmax = 0;
        kmin = 0;
        kmax = 0;
        area_num = 0;
        pk = 0
        pls = NaN
        Ep = 0;
        Ed = 0;
        No = 0;
        % variables - pilots
        pil_val = NaN;
        area_len = 0;
        rho = 0;        % the probability that the signal is 
        thres = 0;      % the threshold that this point is not noise (in power)
    end

    methods
        %{
        @oc: OTFS configuration
        @lmax: the maximal delay index
        @kmax: the maximal Doppler index
        @No: the noise power
        %}
        function self = CPE(oc, lmax, kmax, Ed, No)
            self.oc = oc;
            self.lmax = lmax;
            self.area_len = lmax + 1;
            self.area_num = floor(oc.L / self.area_len);
            if self.area_num <= 0
                error("There is no space for pilots.");
            end
            % calculate the Doppler positions
            self.pk = floor((self.oc.K - 1)/2);
            % update the k (Doppler) range
            if self.pk - self.kmax > 0
                self.kmin = -kmax;
            else
                self.kmin = -self.pk;
            end
            if self.pk + kmax < self.oc.K
                self.kmax = kmax;
            else
                self.kmax = self.oc.K - self.pk;
            end
            % calculate the delay positions
            self.pls = (0:self.area_num-1)*(lmax + 1);
            % calculate the threshold
            self.Ed = Ed;
            self.No = No;
            self.rho = chi2inv(0.9999, 2*self.area_num)/(2*self.area_num);
            self.thres = self.rho*(Ed + No);
        end

        %{
        get the Doppler range
        %}
        function [kmin, kmax] = getKRange(self)
            kmin = self.kmin;
            kmax = self.kmax;
        end

        %{
        generate pilots
        @Ep: the pilot energy
        %}
        function Xp = genPilots(self, Ep)
            self.Ep = Ep;
            self.pil_val = self.PIL_VAL*sqrt(Ep);
            Xp = zeros(self.oc.K, self.oc.L);
            Xp(self.pk+1, self.pls+1) = self.pil_val;       % matlab index starts from 1
        end

        %{
        estimate the path (h, k, l)
        @Y_DD:              the received resrouce grid (N, M)
        @is_all(opt):       estimate paths on all locations
        @est_type(opt):     the estimation type
        %} 
        function varargout = estPaths(self, Y_DD, varargin)
            % load optional inputs 
            inPar = inputParser;
            addParameter(inPar,"is_all", false, @(x) isscalar(x)&&islogical(x));
            addParameter(inPar,"est_type", self.EST_TYPE_LS, @(x) isscalar(x)&&isnumeric(x)&&ismember(x, self.EST_TYPES));
            parse(inPar, varargin{:});
            is_all = inPar.Results.is_all;
            est_type = inPar.Results.est_type;
            % input check
            if ~ismember(est_type, self.EST_TYPES)
                error("CHE type is not supported.");
            end

            % we sum all area together
            est_area = zeros(self.oc.K, self.area_len);
            ca_id_beg = 1;
            ca_id_end = self.area_len;
            % accumulate all areas together
            for area_id = 1:self.area_num
                % build the phase matrix
                est_area = est_area + Y_DD(:, ca_id_beg:ca_id_end).*self.getPhaseConjMat(area_id);
                ca_id_beg = ca_id_end + 1;
                ca_id_end = ca_id_end + self.area_len;
            end
            est_area = est_area/self.area_num;
            % locate the searching space for ki
            ki_beg = self.pk + self.kmin; 
            ki_end = self.pk + self.kmax;
            % find paths
            pmax = (self.kmax - self.kmin + 1)*self.area_len;
            his = zeros(pmax, 1);
            his_var = zeros(pmax, 1);    % estimation error
            his_mask = false(pmax, 1);
            kis = zeros(pmax, 1);
            lis = zeros(pmax, 1);
            pid = 0;
            for l_id = 0:self.area_len-1
                for k_id = ki_beg:ki_end
                    pss_ys = est_area(k_id+1, l_id+1);
                    if is_all || abs(pss_ys)^2 > self.thres
                        % find a path
                        pid = pid + 1;
                        % estimate the channel
                        if est_type == self.EST_TYPE_LS
                            hi = pss_ys/self.pil_val;
                            hi_var = repmat(self.Ed/self.Ep + self.No/self.Ep, size(pss_ys));
                        end
                        % zero values under the threshold
                        if abs(pss_ys)^2 <= self.thres
                            hi = 0;
                            hi_var = eps;
                        end
                        hi_mask = abs(pss_ys)^2 > self.thres;
                        li = l_id - self.pls(1);
                        ki = k_id - self.pk;
                        % append
                        his(pid) = hi;     
                        his_var(pid) = hi_var;
                        his_mask(pid) = hi_mask;
                        lis(pid) = li;
                        kis(pid) = ki;
                    end
                end
            end
            % cut the residual data
            if ~is_all
                his = his(1:pid);
                his_var = his_var(1:pid);
                lis = lis(1:pid);
                kis = kis(1:pid);
            end
            % return
            varargout{1} = his;
            varargout{2} = his_var;
            if is_all
                varargout{3} = his_mask;
            else
                varargout{3} = lis;
                varargout{4} = kis;
            end
        end

    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % private methods
    methods(Access=private)
        %{
        build the phase Conj matrix
        @area_id: the area index
        %}
        function phaseConjMat = getPhaseConjMat(self, area_id)
            if area_id < 1 || area_id > self.area_num
                error("Area Id is illegal.");
            end
            phaseConjMat = zeros(self.oc.K, self.area_len);
            for ki = 0:self.oc.K-1
                for li = 0:self.area_len-1
                    if self.oc.isPulIdeal()
                        phaseConjMat(ki+1, li+1) = exp(2j*pi*li*(ki-self.pk)/self.oc.L/self.oc.K);
                    elseif self.oc.isPulRecta()
                        phaseConjMat(ki+1, li+1) = exp(-2j*pi*self.pls(area_id)*(ki-self.pk)/self.oc.L/self.oc.K);
                    else
                        error("The pulse type is unkownn.");
                    end
                end
            end       
        end
    end
end