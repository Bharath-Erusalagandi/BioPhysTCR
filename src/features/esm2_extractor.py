"""ESM2 Embedding Extractor for BioPhysTCR"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
import esm


class ESM2Extractor:
    """Extract ESM2 embeddings for protein sequences."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        batch_size: int = 32,
        repr_layer: int = 33,
    ):
        """Initialize ESM2 extractor."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.repr_layer = repr_layer

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading ESM2 model: {model_name}")
        print(f"Using device: {self.device}")

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()

        print(f"ESM2 model loaded successfully")

    @torch.no_grad()
    def extract_single(self, sequence: str) -> np.ndarray:
        """Extract embedding for a single sequence."""
        data = [("seq", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)

        representations = results["representations"][self.repr_layer]
        seq_repr = representations[0, 1:len(sequence)+1, :].cpu().numpy()

        return seq_repr

    @torch.no_grad()
    def extract_batch(
        self,
        sequences: List[str],
        pooling: str = "mean"
    ) -> np.ndarray:
        """Extract embeddings for a batch of sequences."""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        representations = results["representations"][self.repr_layer]

        embeddings = []
        for i, seq in enumerate(sequences):
            seq_repr = representations[i, 1:len(seq)+1, :].cpu().numpy()

            if pooling == "mean":
                embeddings.append(seq_repr.mean(axis=0))
            elif pooling == "max":
                embeddings.append(seq_repr.max(axis=0))
            elif pooling == "cls":
                embeddings.append(seq_repr[0])
            else:
                embeddings.append(seq_repr)

        if pooling != "none":
            return np.stack(embeddings)
        return embeddings

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        cdr3_col: str = "CDR3",
        epitope_col: str = "Epitope",
        pooling: str = "mean",
        save_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings for all sequences in a dataframe."""
        unique_cdr3 = df[cdr3_col].unique().tolist()
        unique_epitopes = df[epitope_col].unique().tolist()

        print(f"Extracting embeddings for {len(unique_cdr3)} unique CDR3 sequences")
        print(f"Extracting embeddings for {len(unique_epitopes)} unique epitope sequences")

        cdr3_embeddings = {}
        for i in tqdm(range(0, len(unique_cdr3), self.batch_size), desc="CDR3 embeddings"):
            batch = unique_cdr3[i:i+self.batch_size]
            batch_emb = self.extract_batch(batch, pooling=pooling)
            for seq, emb in zip(batch, batch_emb):
                cdr3_embeddings[seq] = emb

        epitope_embeddings = {}
        for i in tqdm(range(0, len(unique_epitopes), self.batch_size), desc="Epitope embeddings"):
            batch = unique_epitopes[i:i+self.batch_size]
            batch_emb = self.extract_batch(batch, pooling=pooling)
            for seq, emb in zip(batch, batch_emb):
                epitope_embeddings[seq] = emb

        result = {
            "cdr3": cdr3_embeddings,
            "epitope": epitope_embeddings,
        }

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Saved embeddings to {save_path}")

        return result

    def extract_paired_embeddings(
        self,
        df: pd.DataFrame,
        cdr3_col: str = "CDR3",
        epitope_col: str = "Epitope",
        pooling: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract paired embeddings maintaining row correspondence."""
        embeddings = self.extract_from_dataframe(df, cdr3_col, epitope_col, pooling)

        cdr3_emb = np.stack([embeddings["cdr3"][seq] for seq in df[cdr3_col]])
        epitope_emb = np.stack([embeddings["epitope"][seq] for seq in df[epitope_col]])

        return cdr3_emb, epitope_emb


def load_esm2_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load pre-computed ESM2 embeddings from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ESM2 embeddings")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with sequences")
    parser.add_argument("--output_path", type=str, required=True, help="Output pickle file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls", "none"])

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} sequences from {args.data_path}")

    extractor = ESM2Extractor(batch_size=args.batch_size)
    extractor.extract_from_dataframe(
        df,
        pooling=args.pooling,
        save_path=Path(args.output_path)
    )
