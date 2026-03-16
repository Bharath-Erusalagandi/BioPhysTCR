import pandas as pd

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'

try:
    xl = pd.ExcelFile(excel_path)
    df = pd.read_excel(excel_path, sheet_name=xl.sheet_names[1])
    
    nan_virus = df[df['Virus Diseases'].isna()]
    print(f"Number of NaN Virus rows: {len(nan_virus)}")
    if len(nan_virus) > 0:
        print("Datasets for NaN Virus rows:")
        print(nan_virus['Dataset'].value_counts())
        
        print("\nSample names for first 5 NaN rows:")
        print(nan_virus['sample_name'].head())
        
except Exception as e:
    print(e)
