classdef VB < Modu
    properties
        constel {mustBeNumeric}
        constel_len {mustBeNumeric}
        Ed = 1;                                             % energy of data (constellation average power)
    end

    methods
        %{
        constructor
        %}
        function self = VB(modu, frame, pul, nTimeslot, nSubcarr, varargin)
            self = self@Modu(modu, frame, pul, nTimeslot, nSubcarr, varargin{:});
        end


        %{
        channel estimation
        @Ydd:           Rx in the DD domain
        <OPT>
        @No:            the noise power
        @Eh:            the power of a path
        @min_var:       the minimal variance.
        @iter_num:      the maximal iteration
        @es:            early stop
        @es_thres:      early stop threshold (abs)
        %}
        function h_mean = che(self, Ydd, varargin)
            if self.modu ~= self.MODU_OTFS_EMBED
                error("CHE is not supported for non-embedded OTFS!!!");
            end
            
            % load optional inputs 
            inPar = inputParser;
            addParameter(inPar, "No",         NaN);
            addParameter(inPar, "Eh",         NaN);
            addParameter(inPar, "min_var",    eps);
            addParameter(inPar, "iter_num",   125);
            addParameter(inPar, "es",  true);
            addParameter(inPar, "es_thres",  1e-6);
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            No          = inPar.Results.No;
            Eh          = inPar.Results.Eh;
            min_var     = inPar.Results.min_var;
            iter_num    = inPar.Results.iter_num;
            es          = inPar.Results.es;
            es_thres    = inPar.Results.es_thres;

            % init parameters
            Yp = Ydd(self.pilCheRng(1):self.pilCheRng(2), self.pilCheRng(3):self.pilCheRng(4));
            yp = Yp(:);
            Z = length(yp);
            P = self.ref2Phi();
            PtP = P'*P;
            Pty = P'*yp;
            a = 1; b = 1;
            c = ones(self.pmax, 1);
            d = ones(self.pmax, 1);
            
            
            gamma = ones(self.pmax, 1);
            h_vari = inv(PtP + eye(self.pmax));
            h_mean = h_vari*Pty;
            % alpha
            alpha = 1;
            update_alpha = false;
            if ~isnan(No)
                alpha = 1/No;
            else
                update_alpha = true;
            end
            % beta
            beta_mean = zeros(self.pmax, 1);
            beta_vari = ones(self.pmax, 1);
            update_beta = false;
            if ~isnan(Eh)
                beta_vari = Eh*beta_vari;
                update_beta = true;
            end
            
            % VB CHE
            for t = 1:iter_num
                % update alpha
                if update_alpha
                    a = a + Z;
                    b = b + sum(yp - P*h_mean) + sum(diag(PtP*h_vari));
                    alpha = a/b;
                end

                % update h
                h_vari = inv(alpha*PtP + diag(gamma));
                %h_mean = h_vari*alpha*Pty;
                h_mean = h_vari*(alpha*Pty + gamma.*beta_mean);
                % update gamma
                c = c + 1;
                %d = d + real(diag(h_vari)) + abs(h_mean).^2;
                d = d + real(diag(h_vari)) + abs(h_mean).^2 - conj(beta_mean).*h_mean - conj(h_mean).*beta_mean + real(beta_vari) + abs(beta_mean).^2;
                gamma_new = c./d;
                % update beta
                if update_beta
                    beta_vari = 1./(gamma + beta_vari.^(-1));
                    beta_mean = beta_vari.*(gamma.*h_mean + beta_mean./beta_vari);
                end

                % early stop
                if es
                    if sum(abs(gamma_new - gamma).^2)/sum(abs(gamma).^2) < es_thres
                        break;
                    end
                end
                gamma = gamma_new;
                
            end
            

        end
    end
end