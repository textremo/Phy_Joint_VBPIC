classdef VB < Modu
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
            addParameter(inPar, "min_var",    eps);
            addParameter(inPar, "iter_num",   125);
            addParameter(inPar, "es",  true);
            addParameter(inPar, "es_thres",  1e-6);
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            No          = inPar.Results.No;
            min_var     = inPar.Results.min_var;
            iter_num    = inPar.Results.iter_num;
            es          = inPar.Results.es;
            es_thres    = inPar.Results.es_thres;

            % init parameters
            Yp = Ydd(self.pilCheRng(1):self.pilCheRng(2), self.pilCheRng(3):self.pilCheRng(4));
            yp = Yp(:);
            P = self.ref2Phi();
            PtP = P'*P;
            Pty = P'*yp;
            a = 1; b = 1; c = ones(self.pmax, 1); d = ones(self.pmax, 1);
            alpha = 1; gamma = ones(self.pmax, 1); h_vari = inv(PtP + eye(self.pmax)); h_mean = h_vari*Pty;
            update_alpha = false;
            if ~isnan(No)
                alpha = 1/No;
            else
                alpha = 1;
                update_alpha = true;
            end
            
            % VB CHE
            for t = 1:iter_num
                % update alpha
                if update_alpha
                end

                % update h
                h_vari = inv(alpha*PtP + diag(gamma));
                h_mean = alpha*h_vari*Pty;
                % update gamma
                c = c + 1;
                d = d + diag(h_vari) + abs(h_mean).^2;
                gamma_new = c./d;

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