import inspect
import os
import pandas as pd


def genconfig(filename, sheetname, casename):
    # callers globel
    cgs = inspect.currentframe().f_back.f_globals
    
    # Specify the file path and sheet number
    filename = filename + ".xlsx";
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
       
    sheet = pd.read_excel(io=filename, 
                          sheet_name=sheetname, 
                          engine='openpyxl',
                          dtype=str)
    for i, row in sheet.iterrows():
        props_name = row.iloc[0]
        prop = row[casename]
        
        cgs[props_name] = eval(prop)