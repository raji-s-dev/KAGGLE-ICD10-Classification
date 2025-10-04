#!/usr/bin/env python3
"""
optimize_thresholds.py

Two-step:
 1) Run model on X_val to compute per-sample probabilities (probs_val.npy)
 2) Find per-label thresholds (mode: 'per_label' or 'greedy') and save thresholds.npy

Usage: python optimize_thresholds.py --help
"""
import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---- Model (same as train/test) ----
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
    def __init__(self, input_dim, output_dim, hidden_dims=(1024,512,512,256), dropout=0.3):
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

# ---- helper funcs ----
def load_scaler(scaler_path):
    with open(scaler_path, 'r') as f:
        d = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(d["mean"], dtype=np.float32)
    scaler.scale_ = np.array(d["scale"], dtype=np.float32)
    return scaler

def predict_probs(X, model, device, batch_size=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
    return np.vstack(probs)

def micro_f1_from_counts(tp, fp, fn):
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ---- per-label independent optimization (fast) ----
def per_label_search(probs, y_true, grid):
    n_labels = probs.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=float)
    for j in range(n_labels):
        col = probs[:, j]
        yt = y_true[:, j]
        # skip labels with zero positives
        if yt.sum() == 0:
            thresholds[j] = 1.0
            continue
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            pred_col = (col >= t).astype(np.uint8)
            tp = int(((pred_col == 1) & (yt == 1)).sum())
            fp = int(((pred_col == 1) & (yt == 0)).sum())
            fn = int(((pred_col == 0) & (yt == 1)).sum())
            f1 = micro_f1_from_counts(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[j] = best_t
    return thresholds

# ---- greedy coordinate-ascent optimization (optimizes global micro-F1) ----
def greedy_optimize(probs, y_true, init_threshold=0.65, grid=None, max_iters=3, verbose=True, top_k=None):
    n_labels = probs.shape[1]
    if grid is None:
        grid = np.linspace(0.3, 0.7, 41)
    thresholds = np.full(n_labels, init_threshold, dtype=float)
    # initial predictions
    base_preds = (probs >= thresholds)
    positives = y_true.sum(axis=0).astype(int)  # per-label positives
    base_tp = int(((base_preds == 1) & (y_true == 1)).sum())
    base_fp = int(((base_preds == 1) & (y_true == 0)).sum())
    base_fn = int(((base_preds == 0) & (y_true == 1)).sum())

    if top_k is not None:
        # optimize only top_k labels by frequency
        order = np.argsort(-positives)[:top_k]
    else:
        # default order: descending frequency (more stable), but we iterate all labels
        order = np.argsort(-positives)

    if verbose:
        print(f"Starting greedy optimization: init_thresh={init_threshold} labels={n_labels} top_k={top_k}")

    it = 0
    improved_global = True
    while it < max_iters and improved_global:
        it += 1
        improved_global = False
        if verbose:
            print(f"\nIteration {it}/{max_iters}")
        for j in order:
            col_probs = probs[:, j]
            y_col = y_true[:, j]
            base_col_pred = base_preds[:, j].astype(bool)
            base_tp_j = int(((base_col_pred == True) & (y_col == 1)).sum())
            base_fp_j = int(((base_col_pred == True) & (y_col == 0)).sum())
            base_fn_j = int(((base_col_pred == False) & (y_col == 1)).sum())
            best_local_t = thresholds[j]
            best_local_score = micro_f1_from_counts(base_tp, base_fp, base_fn)  # start from current global
            # try each candidate threshold
            for t in grid:
                cand = (col_probs >= t)
                tp_j_new = int(((cand == True) & (y_col == 1)).sum())
                fp_j_new = int(((cand == True) & (y_col == 0)).sum())
                fn_j_new = int(((cand == False) & (y_col == 1)).sum())
                tp_new = base_tp - base_tp_j + tp_j_new
                fp_new = base_fp - base_fp_j + fp_j_new
                fn_new = base_fn - base_fn_j + fn_j_new
                score = micro_f1_from_counts(tp_new, fp_new, fn_new)
                if score > best_local_score + 1e-12:
                    best_local_score = score
                    best_local_t = t
            # if improved, update
            if best_local_t != thresholds[j]:
                # compute new base counts and base_preds for updates
                cand_pred = (col_probs >= best_local_t)
                new_tp_j = int(((cand_pred == True) & (y_col == 1)).sum())
                new_fp_j = int(((cand_pred == True) & (y_col == 0)).sum())
                new_fn_j = int(((cand_pred == False) & (y_col == 1)).sum())
                base_tp = base_tp - base_tp_j + new_tp_j
                base_fp = base_fp - base_fp_j + new_fp_j
                base_fn = base_fn - base_fn_j + new_fn_j
                base_preds[:, j] = cand_pred
                thresholds[j] = best_local_t
                improved_global = True
                if verbose and (j % 100 == 0):
                    print(f" label {j} improved -> t={best_local_t:.3f} new_global_f1={micro_f1_from_counts(base_tp, base_fp, base_fn):.6f}")
        if verbose:
            print(f"End of iter {it}. improved_global={improved_global}. global_f1={micro_f1_from_counts(base_tp, base_fp, base_fn):.6f}")
    return thresholds, micro_f1_from_counts(base_tp, base_fp, base_fn)

# ---- main CLI ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--X-val", required=True)
    p.add_argument("--Y-val", required=True)
    p.add_argument("--model-path", required=False, default=None)
    p.add_argument("--probs-path", required=False, default=None, help="If provided, skip model inference")
    p.add_argument("--scaler-path", required=False, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[1024,512,512,256])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--mode", choices=["per_label","greedy"], default="greedy")
    p.add_argument("--init-threshold", type=float, default=0.65)
    p.add_argument("--grid-start", type=float, default=0.30)
    p.add_argument("--grid-end", type=float, default=0.70)
    p.add_argument("--grid-steps", type=int, default=41)
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--top-k", type=int, default=None, help="Only optimize top-k frequent labels (speed)")
    p.add_argument("--save-probs", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    X_val = np.load(args.X_val)
    Y_val = np.load(args.Y_val).astype(np.uint8)
    print(f"Loaded X_val {X_val.shape}, Y_val {Y_val.shape}")

    # get probabilities (either load or run model)
    if args.probs_path and os.path.exists(args.probs_path):
        print(f"Loading probs from {args.probs_path}")
        probs = np.load(args.probs_path)
    else:
        assert args.model_path is not None, "Provide model-path to run inference"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # scale X_val if scaler provided
        if args.scaler_path and os.path.exists(args.scaler_path):
            scaler = load_scaler(args.scaler_path)
            X_val_scaled = scaler.transform(X_val)
            print("Applied scaler to X_val")
        else:
            X_val_scaled = X_val
            print("No scaler provided, using raw X_val")
        input_dim = X_val_scaled.shape[1]
        output_dim = Y_val.shape[1]
        model = DeepResidualMLP(input_dim, output_dim, hidden_dims=tuple(args.hidden_dims), dropout=args.dropout).to(device)
        ckpt = torch.load(args.model_path, map_location=device)
        # support both styles: either ckpt contains 'model_state' or plain state_dict
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded model from {args.model_path}")
        probs = predict_probs(X_val_scaled, model, device, batch_size=args.batch_size)
        if args.save_probs:
            np.save(out_dir / "probs_val.npy", probs)
            print(f"Saved probs to {out_dir / 'probs_val.npy'}")

    # threshold grid
    grid = np.linspace(args.grid_start, args.grid_end, args.grid_steps)

    if args.mode == "per_label":
        print("Running per-label independent search (fast)...")
        thresholds = per_label_search(probs, Y_val, grid)
        final_preds = (probs >= thresholds).astype(np.uint8)
        # compute global micro f1
        tp = int(((final_preds == 1) & (Y_val == 1)).sum())
        fp = int(((final_preds == 1) & (Y_val == 0)).sum())
        fn = int(((final_preds == 0) & (Y_val == 1)).sum())
        final_f1 = micro_f1_from_counts(tp, fp, fn)
        print(f"Per-label search done. micro-F1 = {final_f1:.6f}")
    else:
        print("Running greedy coordinate-ascent (optimizes micro-F1)...")
        thresholds, final_f1 = greedy_optimize(
            probs, Y_val, init_threshold=args.init_threshold,
            grid=grid, max_iters=args.max_iters, verbose=True, top_k=args.top_k
        )
        print(f"Greedy optimization done. micro-F1 = {final_f1:.6f}")

    # Save thresholds and meta
    np.save(out_dir / "thresholds.npy", thresholds)
    meta = {
        "mode": args.mode,
        "init_threshold": args.init_threshold,
        "grid": [float(args.grid_start), float(args.grid_end), int(args.grid_steps)],
        "max_iters": args.max_iters,
        "final_micro_f1": float(final_f1),
        "n_labels": int(probs.shape[1]),
    }
    with open(out_dir / "thresholds_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved thresholds to {out_dir / 'thresholds.npy'} and meta to thresholds_meta.json")

if __name__ == "__main__":
    main()
