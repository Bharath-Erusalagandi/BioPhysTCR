import pandas as pd

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'

try:
    df = pd.read_excel(excel_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for "exposed to or infected with the SARSCoV-2 virus"
    # The user mentioned "additional 72 subjects exposed to or infected with the SARSCoV-2 virus"
    # I'll look for columns related to disease or condition
    possible_cols = [c for c in df.columns if 'disease' in c.lower() or 'condition' in c.lower() or 'virus' in c.lower() or 'diag' in c.lower()]
    print("\nPossible relevant columns:", possible_cols)
    
    if possible_cols:
         for col in possible_cols:
             print(f"\nValue counts for {col}:")
             print(df[col].value_counts().head())

except Exception as e:
    print(f"Error reading excel: {e}")
