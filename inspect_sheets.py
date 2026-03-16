import pandas as pd

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'

try:
    xl = pd.ExcelFile(excel_path)
    print("Sheet names:", xl.sheet_names)
    
    for sheet in xl.sheet_names:
        print(f"\n--- Head of sheet: {sheet} ---")
        df = pd.read_excel(excel_path, sheet_name=sheet, nrows=5)
        print(df.columns.tolist())
        print(df.head())
        
except Exception as e:
    print(f"Error: {e}")
