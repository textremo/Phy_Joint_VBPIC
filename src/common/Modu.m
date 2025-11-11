classdef Modu < dynamicprops
    properties(Constant)
        % OFDM 0~49
        MODU_OFDM               = 0;
        % OTFS 50~99
        MODU_OTFS_FULL          = 50;           % full
        MODU_OTFS_EMBED         = 51;           % embed
        MODU_OTFS_SP            = 60;           % superimposed
        MODU_OTFS_SP_REP_DELAY  = 65;           % superimposed - replicate on the delay axis
        MODUs = [Modu.MODU_OFDM, ... 
                 Modu.MODU_OTFS_FULL, ...
                 Modu.MODU_OTFS_EMBED, ...
                 Modu.MODU_OTFS_SP, ...
                 Modu.MODU_OTFS_SP_REP_DELAY];

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
        @dataLocs:       data locations, a 01 matrix of [N, M] or [K, L]
        <OPT>
        @refSig:         the reference sigal of [N, M] or [K, L], 0 at non-ref locations
        @csiLim:         CSI limitation
                         1)
                         2) OTFS: [lmax, kmax]
        %}
        function self = Modu(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin)
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
            self.dataLocs = dataLocs;
            self.sig_len = nTimeslot*nSubcarr;
            self.data_len = sum(dataLocs, "all");
            if length(varargin) >= 1
                self.refSig = varargin{1};
                if modu == self.MODU_OTFS_FULL
                    error("Full data does not support any reference signal!!!");
                end
            end
            if length(varargin) >= 2
                self.csiLim = varargin{2};
            end
            
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
end