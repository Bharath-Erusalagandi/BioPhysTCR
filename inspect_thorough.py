import pandas as pd
import tarfile
import os

excel_path = 'ImmuneCODE-Repertoire-Tags-002.2.xlsx'
tar_path = 'ImmuneCODE-Repertoires-002.2.tgz'

print("--- Inspecting Excel ---")
try:
    df = pd.read_excel(excel_path)
    print("All Columns:", list(df.columns))
    
    # Check for specific keywords
    keywords = ['epitope', 'antigen', 'target', 'pep', 'bind', 'covid', 'sars', 'virus', 'diagnosis']
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            print(f"Found relevant column: {col}")
            print(df[col].value_counts().head())

    # Filter for COVID-19
    # The user mentioned "additional 72 subjects exposed to or infected with the SARSCoV-2 virus"
    # Looking for 'Virus Diseases' based on previous output or similar
    if 'Virus Diseases' in df.columns:
        covid_df = df[df['Virus Diseases'].astype(str).str.contains('COVID|SARS', case=False, na=False)]
        print(f"\nFound {len(covid_df)} rows with COVID/SARS in 'Virus Diseases'")
        if not covid_df.empty:
            print("Sample IDs of COVID patients:", covid_df['sample_name'].head().tolist())
    
except Exception as e:
    print(f"Error reading excel: {e}")

print("\n--- Inspecting One TSV from Tar ---")
try:
    with tarfile.open(tar_path, "r:gz") as tar:
        # Find a TSV
        for member in tar:
            if member.name.endswith('.tsv'):
                f = tar.extractfile(member)
                if f:
                    content = pd.read_csv(f, sep='\t', nrows=5)
                    print(f"Columns in {member.name}:", list(content.columns))
                    print(content.head())
                break
except Exception as e:
    print(f"Error reading tar: {e}")
