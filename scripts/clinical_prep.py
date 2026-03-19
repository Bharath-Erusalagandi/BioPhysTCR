
import pandas as pd
import tarfile
import os
import argparse
from pathlib import Path
import random
from tqdm import tqdm

# Standard Immunodominant SARS-CoV-2 Epitopes (Class I)
COVID_EPITOPES = [
    "YLQPRTFLL", # Spike
    "RLQSLQTYV", # Spike (S)
    "KLPDDFTGCV", # Spike
    "NYNYLYRLF", # Spike
    "QYIKWPWYI", # Spike
    "SPRWYFYYL", # Nucleocapsid
]

# Control Epitopes (Flu, CMV, EBV)
CONTROL_EPITOPES = [
    "GILGFVFTL", # Flu M1
    "FMYSDFHFI", # Flu basic
    "NLVPMVATV", # CMV pp65
    "GLCTLVAML", # EBV BMLF1
    "ELAGIGILTV", # Melan-A (Cancer - distinct)
    "IVTDFSVIK", # Ebola (Distinct)
]

def load_metadata(excel_path):
    print(f"Loading metadata from {excel_path}...")
    try:
        df = pd.read_excel(excel_path, sheet_name='All Tags')
    except:
        xl = pd.ExcelFile(excel_path)
        df = pd.read_excel(excel_path, sheet_name=xl.sheet_names[1])
    print(f"Loaded {len(df)} rows.")
    return df

def identify_covid_cohort(df):
    """Identify ~72 COVID-19 subjects."""
    covid_mask = (df['Dataset'] == 'COVID-19-Adaptive-MIRAMatched') | (df['Virus Diseases'] == 'COVID-19 Positive')
    covid_df = df[covid_mask].copy()
    
    if len(covid_df[covid_df['Dataset'] == 'COVID-19-Adaptive-MIRAMatched']) >= 70:
        covid_df = covid_df[covid_df['Dataset'] == 'COVID-19-Adaptive-MIRAMatched']
        
    print(f"Identified {len(covid_df)} COVID-19 subjects.")
    # Return samples
    samples = {}
    for _, row in covid_df.iterrows():
        samples[row['sample_name']] = 'COVID'
    return samples

def extract_tcrs(tar_path, samples, max_tcrs=50):
    """Extract top N abundant TCRs for given samples."""
    tcrs = [] 
    sample_names = set(samples.keys())
    
    print(f"Extracting TCRs from {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar):
            if not member.isfile() or not member.name.endswith('.tsv'):
                continue
                
            fname_base = os.path.basename(member.name).replace('.tsv', '')
            found_sample = None
            
            if fname_base in sample_names:
                found_sample = fname_base
            else:
                for s in sample_names:
                    if s in fname_base: 
                        found_sample = s
                        break
            
            if found_sample:
                try:
                    f = tar.extractfile(member)
                    df_tsv = pd.read_csv(f, sep='\t', usecols=['amino_acid', 'templates'])
                    df_tsv = df_tsv.dropna(subset=['amino_acid'])
                    df_tsv = df_tsv[df_tsv['amino_acid'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$')]
                    df_tsv = df_tsv[df_tsv['amino_acid'].str.len().between(10, 20)] # Filter reasonable length
                    
                    if 'templates' in df_tsv.columns:
                        df_tsv = df_tsv.sort_values('templates', ascending=False)
                    
                    top_tcrs = df_tsv.head(max_tcrs)['amino_acid'].tolist()
                    
                    for seq in top_tcrs:
                        tcrs.append({
                            'cdr3': seq,
                            'cohort': samples[found_sample],
                            'sample': found_sample
                        })
                        
                except Exception as e:
                    print(f"Error processing {member.name}: {e}")

    print(f"Extracted {len(tcrs)} unique TCRs.")
    return tcrs

def create_dataset(tcrs, output_path):
    """Create pairs:"""
    pairs = []
    
    print("Generating pairs...")
    for tcr in tqdm(tcrs):
        # Pick 1 random COVID epitope and 1 random Control epitope per TCR
        # Or test against ALL? 
        # Testing against ALL is better for robust stats (Average Score).
        
        # Test against all 6 COVID epitopes
        for epi in COVID_EPITOPES:
            pairs.append({
                'cdr3': tcr['cdr3'],
                'epitope': epi,
                'pdb_id': '1ao7', # Placeholder
                'label': 1, # "Hypothesized Target"
                'group': 'COVID_Target',
                'sample': tcr['sample']
            })
            
        # Test against all 6 Control epitopes
        for epi in CONTROL_EPITOPES:
            pairs.append({
                'cdr3': tcr['cdr3'],
                'epitope': epi,
                'pdb_id': '1ao7', # Placeholder
                'label': 0, # "Control"
                'group': 'Non_COVID_Control',
                'sample': tcr['sample']
            })
            
    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} pairs to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', required=True)
    parser.add_argument('--tar', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_tcrs', type=int, default=20) # 20 top TCRs per patient * 12 epitopes = 240 pairs/patient * 72 ~ 17k inferences
    args = parser.parse_args()
    
    df_meta = load_metadata(args.excel)
    covid_samples = identify_covid_cohort(df_meta)
    
    tcrs = extract_tcrs(args.tar, covid_samples, max_tcrs=args.max_tcrs)
    create_dataset(tcrs, args.output)

if __name__ == '__main__':
    main()
