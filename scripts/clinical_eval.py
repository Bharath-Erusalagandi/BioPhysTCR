
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from models import GARSEF, GARSEFConfig
from utils import GARSEFDataset, collate_garsef

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config, device='cuda'):
    print(f"Loading model from {checkpoint_path}...")
    model_config = GARSEFConfig(
        esm2_dim=config['model']['sequence']['input_dim'],
        sequence_hidden_dim=config['model']['sequence']['hidden_dim'],
        saprot_dim=config['model']['structure']['input_dim'],
        structure_hidden_dim=config['model']['structure']['hidden_dim'],
        structure_num_gnn_layers=config['model']['structure']['num_layers'],
        physchem_dim=config['model']['physicochemical']['input_dim'],
        physchem_hidden_dim=config['model']['physicochemical']['hidden_dim'],
        fusion_dim=config['model']['fusion']['hidden_dim'],
        dropout=config['model']['sequence']['dropout'],
        fusion_dropout=config['model']['fusion']['dropout'],
        temperature=config['training']['contrastive']['temperature'],
    )
    
    model = GARSEF(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_metadata = []
    
    print("Running inference...")
    for batch in tqdm(dataloader):
        tcr_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['tcr'].items()}
        pmhc_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['pmhc'].items()}
        
        outputs = model(tcr_data, pmhc_data)
        preds = torch.sigmoid(outputs['binding_logits']).squeeze().cpu().numpy()
        
        # Handle single batch item
        if preds.ndim == 0:
            preds = np.expand_dims(preds, 0)
            
        all_preds.extend(preds)
        all_labels.extend(batch['label'].numpy())
        all_metadata.extend(batch['metadata'])
        
    return np.array(all_preds), np.array(all_labels), all_metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_file', default='data/splits/clinical_test_specificity.csv')
    parser.add_argument('--features_dir', default='data/processed_clinical')
    parser.add_argument('--output_dir', default='results/clinical_validation')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Config (assume standard location)
    config = load_config(PROJECT_DIR / 'configs/config.yaml')
    
    # Load Model
    model = load_model(args.checkpoint, config, device)
    
    # Load Data
    dataset = GARSEFDataset(args.data_file, args.features_dir)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_garsef, shuffle=False)
    
    # Inference
    y_pred, y_true, metadata = evaluate(model, loader, device)
    
    # Analysis
    results_df = pd.DataFrame(metadata)
    results_df['score'] = y_pred
    results_df['label'] = y_true # 1=COVID_Target, 0=Control
    
    # Group names
    # Our prep script used 'group' column but GARSEFDataset might not pass it in metadata if not configured?
    # Wait, GARSEFDataset code:
    # 'metadata': {'pdb_id': pdb_id, 'cdr3': cdr3_seq, 'epitope': epitope_seq}
    # It does NOT pass all columns.
    # We need to merge back with original CSV or infer from label.
    # Label 1 is COVID Target, Label 0 is Non-COVID Control.
    
    results_df['type'] = results_df['label'].map({1.0: 'COVID Epitopes', 0.0: 'Control Epitopes'})
    
    # Calculate Metrics (Specificity)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    print(f"\nClinical Specificity Results:")
    print(f"AUROC (Distinguishing COVID vs Control Epitopes): {roc_auc:.4f}")
    print(f"AUPR: {pr_auc:.4f}")
    
    # Save Metrics
    metrics = {
        'auroc': float(roc_auc),
        'aupr': float(pr_auc),
        'n_samples': len(y_true)
    }
    with open(output_dir / 'clinical_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Clinical Specificity: COVID vs Control Epitope Recognition')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'clinical_roc.png', dpi=300)
    plt.close()
    
    # Plot Boxplot Scores
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=results_df, x='type', y='score', palette="Set2")
    plt.title('Predicted Binding Scores for COVID Patients')
    plt.ylabel('Binding Probability')
    plt.savefig(output_dir / 'clinical_scores_boxplot.png', dpi=300)
    plt.close()
    
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()
