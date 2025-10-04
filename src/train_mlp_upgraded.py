"""
Upgraded training script for MLP + Asymmetric Loss (ASL)
Implements Phases 0-7:
- Reproducible seeds
- Saved train/val split (with fallback stratified approach)
- StandardScaler normalization saved/loaded
- Residual deep MLP with LayerNorm + Dropout
- ASL loss (same as before)
- Early stopping, ReduceLROnPlateau, gradient clipping
- Mixed precision optional (AMP)
- Logging to CSV and saving best checkpoint

Usage: python train_mlp_embeddings_upgraded.py --help
"""

import os
import argparse
import random
import json
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# ------------------------
# Utilities / Reproducibility
# ------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


# ------------------------
# Residual MLP with LayerNorm
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # if dims don't match, project residual
        self.project = None
        if dim_in != dim_out:
            self.project = nn.Linear(dim_in, dim_out)

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
# Asymmetric Loss (ASL)
# ------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1 - prob
        if self.clip is not None and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1)
        loss_pos = targets * torch.log(prob_pos.clamp(min=self.eps)) * ((1 - prob_pos) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log(prob_neg.clamp(min=self.eps)) * (prob_pos ** self.gamma_neg)
        loss = - (loss_pos + loss_neg)
        return loss.mean()


# ------------------------
# Training / Validation / Checkpointing
# ------------------------

def save_split(out_dir, X_train, X_val, Y_train, Y_val, scaler=None):
    safe_mkdir(out_dir)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(out_dir, "Y_val.npy"), Y_val)
    if scaler is not None:
        # save scaler mean and scale
        scaler_data = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        with open(os.path.join(out_dir, "scaler.json"), "w") as f:
            json.dump(scaler_data, f)


def load_scaler(scaler_path):
    with open(scaler_path, "r") as f:
        d = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(d["mean"], dtype=np.float32)
    scaler.scale_ = np.array(d["scale"], dtype=np.float32)
    return scaler


def write_log_csv(path, header, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def train_model(
    X, Y, out_dir,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    weight_decay=0.0,
    seed=42,
    val_size=0.1,
    use_amp=False,
    patience=6,
    gamma_pos=0,
    gamma_neg=4,
    hidden_dims=(1024, 512, 512, 256),
    dropout=0.3,
    save_every_epoch=False,
    normalize=True
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_mkdir(out_dir)

    # ----------------
    # Split (try iterative split fallback)
    # ----------------
    # Use sklearn's train_test_split for splitting
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state=seed)

    # Save raw splits (before scaling)
    save_split(os.path.join(out_dir, "splits_raw"), X_train, X_val, Y_train, Y_val)

    # ----------------
    # Preprocessing / Scaler
    # ----------------
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        save_split(os.path.join(out_dir, "splits_scaled"), X_train, X_val, Y_train, Y_val, scaler=scaler)
    else:
        save_split(os.path.join(out_dir, "splits_scaled"), X_train, X_val, Y_train, Y_val)

    # Convert to tensors (float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, Y_val_t),
        batch_size=batch_size, shuffle=False
    )

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    model = DeepResidualMLP(input_dim, output_dim, hidden_dims, dropout=dropout).to(device)

    criterion = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience//2, verbose=True)

    scaler_amp = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    log_path = os.path.join(out_dir, "training_log.csv")
    header = ["epoch", "train_loss", "val_loss", "f1_micro", "f1_macro", "lr"]

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                # gradient clipping
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append((probs > 0.5).astype(int))
                trues.append(yb.cpu().numpy())

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        f1_micro = f1_score(trues, preds, average="micro", zero_division=0)
        f1_macro = f1_score(trues, preds, average="macro", zero_division=0)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {np.mean(val_losses):.6f} | F1 (micro): {f1_micro:.4f} | F1 (macro): {f1_macro:.4f} | LR: {cur_lr:.2e}")

        write_log_csv(log_path, header, [epoch, np.mean(train_losses), np.mean(val_losses), f1_micro, f1_macro, cur_lr])

        # Scheduler step (monitor F1 micro)
        scheduler.step(f1_micro)

        # Checkpoint best
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_path = os.path.join(out_dir, "best_mlp.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'f1_micro': f1_micro
            }, ckpt_path)
            print(f"--> Saved best model (f1_micro={f1_micro:.4f}) to {ckpt_path}")
        else:
            epochs_no_improve += 1

        # Optionally save every epoch
        if save_every_epoch:
            epoch_ckpt = os.path.join(out_dir, f"mlp_epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_ckpt)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement in {patience} epochs. Best epoch: {best_epoch} (f1_micro={best_f1:.4f})")
            break

    # Final save: best model already saved
    print(f"Training finished. Best epoch: {best_epoch} with f1_micro={best_f1:.4f}")

    # Save metadata
    meta = {
        'best_epoch': best_epoch,
        'best_f1_micro': float(best_f1),
        'total_epochs_run': epoch,
        'seed': seed,
        'hidden_dims': list(hidden_dims),
        'dropout': dropout,
        'gamma_pos': gamma_pos,
        'gamma_neg': gamma_neg,
        'normalize': normalize,
        'use_amp': use_amp
    }
    with open(os.path.join(out_dir, 'training_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ------------------------
# CLI
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None, help='path to preprocessed directory containing combined_embeddings.npy and multihot_labels_rebuilt.npy')
    parser.add_argument('--out-dir', type=str, default=None, help='output directory to store models/logs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--use-amp', action='store_true', help='use mixed precision (CUDA only)')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--gamma-pos', type=float, default=0)
    parser.add_argument('--gamma-neg', type=float, default=4)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[1024, 512, 512, 256])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save-every-epoch', action='store_true')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')

    args = parser.parse_args()

    # Default data/out dirs based on repo layout if not provided
    cwd = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(cwd)
    data_dir = args.data_dir or os.path.join(repo_root, 'preprocessed')
    out_dir = args.out_dir or os.path.join(repo_root, 'outputs', 'mlp_upgraded', datetime.now().strftime('%Y%m%d_%H%M%S'))

    X_path = os.path.join(data_dir, 'combined_embeddings.npy')
    Y_path = os.path.join(data_dir, 'multihot_labels_rebuilt.npy')

    assert os.path.exists(X_path), f"Embeddings not found: {X_path}"
    assert os.path.exists(Y_path), f"Labels not found: {Y_path}"

    X = np.load(X_path)
    Y = np.load(Y_path)

    train_model(
        X, Y, out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_size=args.val_size,
        use_amp=args.use_amp,
        patience=args.patience,
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        save_every_epoch=args.save_every_epoch,
        normalize=args.normalize
    )
