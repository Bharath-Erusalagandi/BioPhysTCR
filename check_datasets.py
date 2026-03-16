import pandas as pd

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'

try:
    xl = pd.ExcelFile(excel_path)
    df = pd.read_excel(excel_path, sheet_name=xl.sheet_names[1]) # Assuming sheet 1 is tags details
    
    print("Unique Datasets:")
    print(df['Dataset'].unique())
    
    print("\nUnique Virus Diseases:")
    print(df['Virus Diseases'].unique())
    
except Exception as e:
    print(e)
