%{
% generate the config for the current simulation
%}
function genconfig(filename, sheetname, casename)
    % Specify the file path and sheet number
    filename = './config/' + filename + ".xlsx";
    
    % Read the data from the Excel file
    sheet_names = sheetnames(filename);
    sheet_num = length(sheet_names);
    
    
    % opts = setvartype(opts, 'YourColumnName', 'string');

    is_sheet = false;
    is_case = false;

    for i = 1:sheet_num
        sheet_name = sheet_names(i);
        if sheet_name == sheetname
            is_sheet = true;
            opts = detectImportOptions(filename, 'Sheet', i);
            opts.VariableTypes(:) = {'string'};
            opts.VariableNamingRule = 'preserve';
            sheet_content = readtable(filename, opts);
            if ismember(casename, sheet_content.Properties.VariableNames)
                is_case = true;
                props_name = sheet_content.(1);
                props = sheet_content.(casename);
                props_len = length(props);
                
                for prop_id = 1:props_len
                    assignin('base', props_name(prop_id), eval(props(prop_id)));
                end
            end
            break;
        end

        
    end

    if ~is_sheet
        error("[%s] does not exist!!!", sheetname);
    end
    if ~is_case
        error("[%s] does not exist!!!", casename);
    end
end