"""Evaluation script for BioPhysTCR model."""

import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys

sys.path.append("..")

from src.models.biophystcr import BioPhysTCR, BioPhysTCRConfig
from src.data.dataset import BioPhysTCRDataset
from torch.utils.data import DataLoader


def evaluate_model(model, dataloader, device="cuda"):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            tcr_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["tcr"].items()
            }
            pmhc_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["pmhc"].items()
            }
            labels = batch["label"].to(device)

            outputs = model(tcr_data, pmhc_data)
            preds = torch.sigmoid(outputs["binding_logits"]).cpu().numpy()

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    aupr = average_precision_score(all_labels, all_preds)

    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)
    prec = precision_score(all_labels, binary_preds)
    rec = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)

    return {
        "auc": float(auc),
        "aupr": float(aupr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def main():
    """Main evaluation function."""

    # Load config
    config = BioPhysTCRConfig(
        esm2_dim=1280,
        sequence_hidden_dim=256,
        sequence_num_layers=2,
        saprot_dim=446,
        structure_hidden_dim=256,
        structure_num_gnn_layers=3,
        physchem_dim=8,
        physchem_hidden_dim=64,
        fusion_dim=256,
        projection_dim=128,
        dropout=0.2,
    )

    # Load model
    model = BioPhysTCR(config).cuda()
    checkpoint = torch.load("checkpoints/best_model.pt", map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load validation set (standard benchmark)
    val_dataset = BioPhysTCRDataset("data/processed/val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load zero-shot test set (unseen epitopes)
    zeroshot_dataset = BioPhysTCRDataset("data/processed/test_zeroshot")
    zeroshot_loader = DataLoader(
        zeroshot_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    print("Evaluating on standard validation set...")
    val_metrics = evaluate_model(model, val_loader)

    print("Evaluating on zero-shot test set...")
    zeroshot_metrics = evaluate_model(model, zeroshot_loader)

    # Save results
    results = {
        "standard_benchmark": val_metrics,
        "zero_shot_unseen_epitopes": zeroshot_metrics,
    }

    output_path = Path("results/evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Results ===")
    print(f"\nStandard Benchmark:")
    print(f"  AUC: {val_metrics['auc']:.4f}")
    print(f"  AUPR: {val_metrics['aupr']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

    print(f"\nZero-Shot (Unseen Epitopes):")
    print(f"  AUC: {zeroshot_metrics['auc']:.4f}")
    print(f"  AUPR: {zeroshot_metrics['aupr']:.4f}")
    print(f"  Accuracy: {zeroshot_metrics['accuracy']:.4f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
