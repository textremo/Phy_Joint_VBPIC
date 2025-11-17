%{
% generate the config for the current simulation
%}
function genconfig(filename, sheetname, casename)
    % Specify the file path and sheet number
    filename = './config/' + filename + ".xlsx";
    
    % Read the data from the Excel file
    sheet_names = sheetnames(filename);
    sheet_num = length(sheet_names);
    
    is_sheet = false;

    for i = 1:sheet_num
        sheet_name = sheet_names(i);
        if sheet_name == sheetname
            is_sheet = true;
            sheet_content = readtable(filename, 'Sheet', i, "VariableNamingRule", 'preserve', 'Range', 'A1:Z100');
            break;
        end

        
    end

    if ~is_sheet
        error("[%s] does not exist!!!", sheetname);
    end
end