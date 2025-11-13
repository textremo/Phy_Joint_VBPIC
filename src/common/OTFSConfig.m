% OTFS configuration
classdef OTFSConfig < handle
    % constants
    properties(Constant)
        % frame type
        FRAME_TYPE_GIVEN = -1;                      % user input the frame matrix directly
        FRAME_TYPE_FULL = 0;                        % full data
        FRAME_TYPE_CP = 1;                          % using cyclic prefix between each two adjacent OTFS subframe (no interference between two adjacent OTFS subframes)
        FRAME_TYPE_ZP = 2;                          % using zero padding (no interference between two adjacent OTFS subframes)
        FRAME_TYPES = [OTFSConfig.FRAME_TYPE_GIVEN, OTFSConfig.FRAME_TYPE_FULL, OTFSConfig.FRAME_TYPE_CP, OTFSConfig.FRAME_TYPE_ZP];
        % pulse type
        PUL_TYPE_IDEAL = 10;                        % using ideal pulse to estimate the channel
        PUL_TYPE_RECTA = 20;                        % using rectangular waveform to estimate the channel
        PUL_TYPES = [OTFSConfig.PUL_TYPE_IDEAL, OTFSConfig.PUL_TYPE_RECTA]; 
        % pilot types
        PIL_TYPE_GIVEN = -1;                        % user input the pilot matrix directly
        PIL_TYPE_NO = 0;                            % no pilots
        PIL_TYPE_EM_MID = 10;                       % embedded pilots - middle
        PIL_TYPE_EM_ZP = 11;                        % embedded pilots - zero padding areas
        PIL_TYPE_SP_MID = 20;                       % superimposed pilots - middle
        PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY = 21     % superimposed pilots - multiple orthogonal along delay axis
        PIL_TYPES = [OTFSConfig.PIL_TYPE_GIVEN, ...
            OTFSConfig.PIL_TYPE_NO, ...
            OTFSConfig.PIL_TYPE_EM_MID, ...
            OTFSConfig.PIL_TYPE_EM_ZP, ...
            OTFSConfig.PIL_TYPE_SP_MID, ...
            OTFSConfig.PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY];
    end
    % properties
    properties
        frame_type = OTFSConfig.FRAME_TYPE_FULL;
        pul_type = OTFSConfig.PUL_TYPE_RECTA;
        pil_type = OTFSConfig.PIL_TYPE_NO;
        K = 0;              % timeslot number
        L = 0;              % subcarrier number
        % zero padding length (along delay axis at the end)
        zp_len = 0;
        % pilot
        pk_len = 1;         % length on Dopller axis
        pl_len = 1;         % length on delay axis
        pk_num = 1;         % pilot area (CHE) num on Doppler axis
        pl_num = 1;         % pilot area (CHE) num on delay axis
        % guard lengths on delay Doppler axes (only available for embedded pilots)
        gl_len_neg = 0;
        gl_len_pos = 0;
        gk_len_neg = 0;
        gk_len_pos = 0;
        % energy
        Es_d = 0;           % data energy
        Es_p = 0;           % piloter energy
        % vectorized length
        sig_len = 0;
        data_len = 0;
    end

    methods
        %{
        set the frame
        @frame_type:    frame type
        @K:             timeslote number
        @L:             subcarrier number
        @zp_len:        zero padding length
        %}
        function setFrame(self, frame_type, K, L, varargin)
            % load inputs
            self.frame_type = frame_type;
            self.L = L;
            self.K = K;
            self.sig_len = L*K;
            self.data_len = L*K;
            % load optional inputs 
            inPar = inputParser;
            addParameter(inPar,"zp_len", 0, @(x) isscalar(x)&&isnumeric(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            self.zp_len = inPar.Results.zp_len;
            % input check
            if ~ismember(self.frame_type, self.FRAME_TYPES) 
                error("`frame_type` must be a type of `OTFSConfig.FRAME_TYPES`.")
            elseif self.frame_type == OTFSConfig.FRAME_TYPE_ZP && (self.zp_len <= 0 || self.zp_len >= L)
                error("In zeor padding OTFS, `zp_len` must be positive and less than subcarrier number.");
            end
            if self.L <= 0
                error("Subcarrier number must be positive.");
            end
            if self.K <= 0
                error("Timeslot number must be positive.");
            end
        end

        %{
        set the pulse tye
        @pul_type: pulse type
        %}
        function setPul(self, pul_type)
            if ~ismember(pul_type, OTFSConfig.PUL_TYPES)
                error("`pul_type` must be a type of `OTFSConfig.PUL_TYPES`");
            end
            self.pul_type = pul_type;
        end

        %{
        check the pulse type
        %}
        function chk = isPulIdeal(self)
            chk = self.pul_type == OTFSConfig.PUL_TYPE_IDEAL;
        end
        function chk = isPulRecta(self)
            chk = self.pul_type == OTFSConfig.PUL_TYPE_RECTA;
        end
        
        %{
        set the pilot
        @pil_type: pilote type
        @pk_len: the pilot length along the 
        %}
        function setPil(self, pil_type, varargin) 
            % load inputs
            self.pil_type = pil_type;
            % load optional inputs 
            inPar = inputParser;
            addParameter(inPar,"pk_len", 1, @(x) isscalar(x)&&isnumeric(x));
            addParameter(inPar,"pl_len", 1, @(x) isscalar(x)&&isnumeric(x));
            addParameter(inPar,"pk_num", 1, @(x) isscalar(x)&&isnumeric(x));
            addParameter(inPar,"pl_num", 1, @(x) isscalar(x)&&isnumeric(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            self.pk_len = inPar.Results.pk_len;
            self.pl_len = inPar.Results.pl_len;
            self.pk_num = inPar.Results.pk_num;
            self.pl_num = inPar.Results.pl_num;
            % input check
            if self.frame_type == self.FRAME_TYPE_FULL && (self.pil_type == self.PIL_TYPE_EM_MID || self.pil_type == self.PIL_TYPE_EM_ZP)
                error("The embedded pilots are not allowed to use while the frame is set to full data.");
            end
            if self.frame_type ~= self.FRAME_TYPE_CP && self.pil_type == self.PIL_TYPE_EM_ZP
                error("The zero pading pilots are not allowed to use while the frame is not set to zero padding.");
            end
            % check optional inputs based on pilot type
            if ismember(self.pil_type, [self.PIL_TYPE_EM_MID, self.PIL_TYPE_EM_ZP, self.PIL_TYPE_SP_MID, self.PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY])
                if self.pk_len <= 0 || self.pk_len >= self.K
                    error("`pk_len` must be positive and less than the timeslot number.");
                end
                if self.pl_len <= 0 || self.pl_len >= self.L
                    error("`pl_len` must be positive and less than the subcarrier number.");
                end
            end
            if self.pil_type ==  OTFSConfig.PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY
                if self.pk_num * self.pk_len <= 0 || self.pk_num * self.pk_len >= self.K
                    error("`pk_num` must be positive and short enough to put all pilots along the Doppler axis.");
                end
                if self.pl_num * self.pl_len <= 0 || self.pl_num * self.pk_len >= self.L
                    error("`pk_num` must be positive and short enough to put all pilots along the delay axis.");
                end
            end
        end
            
        %{
        set the guard (only available for embedded pilots)
        %}
        function setGuard(self, gl_len_neg, gl_len_pos, gk_len_neg, gk_len_pos)
            % load inputs
            self.gl_len_neg = gl_len_neg;
            self.gl_len_pos = gl_len_pos;
            self.gk_len_neg = gk_len_neg;
            self.gk_len_pos = gk_len_pos;
            % input check
            if self.pil_type ~= self.PIL_TYPE_EM_MID && self.pil_type ~= self.PIL_TYPE_EM_ZP
                error("The guard is only available when using embedded pilots.");
            end
        end

    end
end