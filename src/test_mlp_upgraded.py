"""
Upgraded testing script for MLP + ASL
- Loads best checkpoint, scaler, and thresholds (optional)
- Applies model to test embeddings
- Saves Kaggle submission CSV (id,labels)

Usage:
    python test_mlp_upgraded.py \
        --test-path preprocessed/test_embeddings.npy \
        --codes-path data/icd_codes.txt \
        --model-path outputs/mlp_upgraded/best_mlp.pt \
        --scaler-path outputs/mlp_upgraded/splits_scaled/scaler.json \
        --out-csv outputs/mlp_upgraded/submission.csv \
        [--global-threshold 0.65]
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ------------------------
# Model (Residual MLP)
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(dim_in, dim_out) if dim_in != dim_out else None

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.project is not None:
            residual = self.project(residual)
        return out + residual

class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512, 512, 256), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(in_dim, h, dropout=dropout))
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, output_dim)
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x

# ------------------------
# Load scaler from JSON
# ------------------------
def load_scaler(scaler_path):
    with open(scaler_path, "r") as f:
        d = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(d["mean"], dtype=np.float32)
    scaler.scale_ = np.array(d["scale"], dtype=np.float32)
    return scaler

# ------------------------
# Inference + Kaggle-style submission
# ------------------------
def run_inference(
    test_path, codes_path, model_path, out_csv,
    scaler_path=None, thresholds_path=None,
    hidden_dims=(1024, 512, 512, 256), dropout=0.3,
    global_threshold=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test embeddings
    X_test = np.load(test_path)
    if scaler_path and os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        scaler = load_scaler(scaler_path)
        X_test = scaler.transform(X_test)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Load ICD codes
    with open(codes_path, "r") as f:
        codes = [line.strip() for line in f]

    input_dim = X_test.shape[1]
    output_dim = len(codes)

    # Build model
    model = DeepResidualMLP(input_dim, output_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded model from {model_path} (epoch={ckpt.get('epoch', '?')}, f1_micro={ckpt.get('f1_micro', '?'):.4f})")

    # ------------------------
    # Thresholds logic
    # ------------------------
    if global_threshold is not None:
        thresholds = np.full(output_dim, global_threshold)
        print(f"⚙️ Using global threshold: {global_threshold}")
    elif thresholds_path and os.path.exists(thresholds_path):
        thresholds = np.load(thresholds_path)
        assert thresholds.shape[0] == output_dim, "Thresholds length must match number of codes"
        print(f"Using per-label thresholds from {thresholds_path}")
    else:
        thresholds = np.full(output_dim, 0.5)
        print("⚙️ Using default threshold: 0.5")

    # Inference
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_t), 256):
            xb = X_test_t[i:i+256].to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    preds = np.vstack(preds)

    # Apply thresholds
    bin_preds = (preds >= thresholds).astype(int)

    # ------------------------
    # Format submission (Kaggle style)
    # ------------------------
    with open(out_csv, "w") as f:
        f.write("id,labels\n")
        for idx, row in enumerate(bin_preds, start=1):
            labels = [codes[j] for j in range(output_dim) if row[j] == 1]
            f.write(f"{idx},{';'.join(labels)}\n")

    print(f"\n✅ Submission CSV saved to: {out_csv}")
    print(f"📊 Total rows in submission: {len(bin_preds)}")

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, required=True, help="path to test embeddings (.npy)")
    parser.add_argument("--codes-path", type=str, required=True, help="path to ICD codes file")
    parser.add_argument("--model-path", type=str, required=True, help="path to trained model checkpoint (.pt)")
    parser.add_argument("--scaler-path", type=str, default=None, help="path to saved scaler.json (optional)")
    parser.add_argument("--thresholds-path", type=str, default=None, help="path to thresholds.npy (optional)")
    parser.add_argument("--out-csv", type=str, required=True, help="path to save submission CSV")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 512, 512, 256])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--global-threshold", type=float, default=None, help="Global threshold for all labels (overrides thresholds.npy)")

    args = parser.parse_args()

    run_inference(
        test_path=args.test_path,
        codes_path=args.codes_path,
        model_path=args.model_path,
        out_csv=args.out_csv,
        scaler_path=args.scaler_path,
        thresholds_path=args.thresholds_path,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        global_threshold=args.global_threshold
    )
