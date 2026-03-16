import pandas as pd

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'

try:
    xl = pd.ExcelFile(excel_path)
    print("Sheet names:", xl.sheet_names)
    
    if len(xl.sheet_names) > 1:
        sheet2 = xl.sheet_names[1]
        print(f"\n--- Inspecting contents of sheet: {sheet2} ---")
        df = pd.read_excel(excel_path, sheet_name=sheet2)
        print("Columns:", list(df.columns))
        print("Head:")
        print(df.head())
        
        # Check for MIRA or Epitope
        for col in df.columns:
            if 'mira' in str(col).lower() or 'epitope' in str(col).lower():
                print(f"Found relevant column: {col}")
                print(df[col].head())
    else:
        print("Only one sheet found.")
        
except Exception as e:
    print(f"Error: {e}")
