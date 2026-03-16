import pandas as pd

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'

try:
    # Read without header to inspect rows
    df = pd.read_excel(excel_path, header=None, nrows=20)
    print("--- First 20 rows of raw Excel read ---")
    print(df)

    # Try to identify header row
    header_row = None
    for i, row in df.iterrows():
        row_values = [str(v).lower() for v in row.values]
        if 'sample_name' in row_values or 'experiment_name' in row_values:
            header_row = i
            print(f"\nFound potential header at row {i}")
            break
    
    if header_row is not None:
        df = pd.read_excel(excel_path, header=header_row)
        print("\n--- Columns with correct header ---")
        print(list(df.columns))
        
        # Check for COVID subjects
        # "Virus Diseases" was seen before.
        # "Tags" usually means clinical tags in ImmuneCODE.
        
        # Check if there is any column with 'Epitope'
        epitope_cols = [c for c in df.columns if 'epitope' in c.lower()]
        print("Epitope columns:", epitope_cols)

        # Check for 'Virus Diseases' values
        if 'Virus Diseases' in df.columns:
            print("\nVirus Diseases (unique values):")
            print(df['Virus Diseases'].unique())
            
            covid_subjects = df[df['Virus Diseases'].astype(str).str.contains("COVID|SARS", case=False, na=False)]
            print(f"\nFound {len(covid_subjects)} COVID subjects.")
            
except Exception as e:
    print(f"Error: {e}")
