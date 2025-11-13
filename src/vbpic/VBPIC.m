classdef VBPIC < Modu
    methods
        function self = VBPIC(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin)
            if ismember(modu, self.MODUS_OFDM)
                
            elseif ismember(modu, self.MODUS_OTFS)
                self = VBPICOTFS(modu, frame, pul, nTimeslot, nSubcarr, dataLocs, varargin{:});
            end
        end
    end
end