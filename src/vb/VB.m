classdef VB < Modu
    methods
        %{
        constructor
        %}
        function self = VB(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin)
            self = self@Modu(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin{:});
        end


        %{
        channel estimation
        <OPT>
        @No:            the noise power
        @min_var:       the minimal variance.
        @iter_num:      the maximal iteration
        @es:            early stop
        @es_thres:      early stop threshold (abs)
        %}
        function che(self, varargin)
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
            a = 1; b = 1; c = 1; d = 1;
            alpha = 1;
            if ~isnan(No)
                alpha = 1/No;
            end

            Phi_p = self.ref2Phi();
        end
    end
end