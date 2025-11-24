#!/usr/bin/env python3
"""
Train on raw observation points (no spatial interpolation).
Supports:
  - legacy single CSV with random train/test split (default, --csv + --val_frac)
  - explicit train/test CSV files (--train_csv and --test_csv) for fixed split (no random split)

Loss: MAE (L1). Outputs eval_per_var.csv and eval_per_day.csv in --out_dir.

cmd
python train.py --train_csv processed_data_mean_train.csv --test_csv processed_data_mean_test.csv --epochs 60 --batch 512 --lr 1e-3 --posenc --posenc_freqs 6 --ckpt_dir checkpoints_nointerp --out_dir eval_nointerp
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils import load_csv
from dataset import RawPointDataset

# -------------------------
# Model: simple MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256, 256, 128], out_dim=4, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Helpers: evaluation + save
# -------------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_trues = []
    all_dates = []
    with torch.no_grad():
        for x, y, dates in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            all_dates.extend(dates)
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    return preds, trues, all_dates

def save_eval_csvs(preds, trues, dates, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # per-var MAE
    mae_per_var = np.mean(np.abs(preds - trues), axis=0)
    count = preds.shape[0]
    import pandas as pd
    df_var = pd.DataFrame({
        'variable': ['so','thetao','uo','vo'],
        'mae': mae_per_var,
        'count': [count]*4
    })
    df_var.to_csv(os.path.join(out_dir, 'eval_per_var.csv'), index=False)

    # per-day aggregation (dates may be ISO strings)
    day_map = defaultdict(list)
    for i, d in enumerate(dates):
        key = d if isinstance(d, str) else (d.isoformat() if hasattr(d, 'isoformat') else str(d))
        day_map[key].append(np.abs(preds[i] - trues[i]))
    rows = []
    for key in sorted(day_map.keys()):
        arr = np.stack(day_map[key], axis=0)
        mae_so = float(arr[:,0].mean())
        mae_thetao = float(arr[:,1].mean())
        mae_uo = float(arr[:,2].mean())
        mae_vo = float(arr[:,3].mean())
        mae_overall = float(np.mean([mae_so, mae_thetao, mae_uo, mae_vo]))
        rows.append({
            'date': key,
            'mae_so': mae_so,
            'mae_thetao': mae_thetao,
            'mae_uo': mae_uo,
            'mae_vo': mae_vo,
            'mae_overall': mae_overall
        })
    df_day = pd.DataFrame(rows, columns=['date','mae_so','mae_thetao','mae_uo','mae_vo','mae_overall'])
    df_day.to_csv(os.path.join(out_dir, 'eval_per_day.csv'), index=False)

# -------------------------
# Training loop
# -------------------------
def train_loop(model, train_loader, val_loader, device, epochs, lr, ckpt_dir):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.L1Loss()
    best_val = float('inf')
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        n = 0
        for x,y,_ in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.shape[0]
            total_loss += loss.item() * bs
            n += bs
        train_mae = total_loss / max(1, n)

        # validation
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for x,y,_ in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                bs = x.shape[0]
                total_loss += loss.item() * bs
                n += bs
        val_mae = total_loss / max(1, n)

        print(f"Epoch {epoch}: train_mae={train_mae:.8e} val_mae={val_mae:.8e}")

        # save best
        if val_mae < best_val:
            best_val = val_mae
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_mae': val_mae},
                       os.path.join(ckpt_dir, 'best_nointerp.pth'))
            print("Saved best checkpoint.")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    # legacy single CSV mode (random split)
    parser.add_argument('--csv', default='processed_data_mean.csv', help='single CSV for random train/test split (legacy)')
    # new explicit train/test mode
    parser.add_argument('--train_csv', default=None, help='CSV file to use as training set (fixed)')
    parser.add_argument('--test_csv', default=None, help='CSV file to use as test set (fixed)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=str, default='256,256,128', help='comma-separated hidden sizes')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--posenc', action='store_true', help='use positional encodings')
    parser.add_argument('--posenc_freqs', type=int, default=6)
    parser.add_argument('--val_frac', type=float, default=0.15, help='used only in single-CSV mode')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints_nointerp')
    parser.add_argument('--out_dir', type=str, default='eval_nointerp')
    args = parser.parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device and pin_memory decision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_mem = True if device.type == 'cuda' else False

    # --------- load data ----------
    if args.train_csv is not None and args.test_csv is not None:
        if args.train_csv == args.test_csv:
            print("Warning: --train_csv and --test_csv are identical. Proceeding but results will be trivial.")
        print(f"Loading fixed train CSV: {args.train_csv}")
        df_train = load_csv(args.train_csv)
        print(f"Loading fixed test CSV: {args.test_csv}")
        df_test = load_csv(args.test_csv)

        dataset_train = RawPointDataset(df_train, use_posenc=args.posenc, posenc_freqs=args.posenc_freqs, scale=True)
        dataset_test  = RawPointDataset(df_test,  use_posenc=args.posenc, posenc_freqs=args.posenc_freqs, scale=True)

        n_train = len(dataset_train)
        n_test = len(dataset_test)
        n_total = n_train + n_test

        train_loader = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=pin_mem)
        test_loader  = DataLoader(dataset_test,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=pin_mem)

    else:
        # legacy single CSV mode with random split
        print(f"Loading single CSV and splitting: {args.csv}")
        df = load_csv(args.csv)
        dataset = RawPointDataset(df, use_posenc=args.posenc, posenc_freqs=args.posenc_freqs, scale=True)
        n_total = len(dataset)
        n_test = max(1, int(n_total * args.val_frac))
        n_train = n_total - n_test
        train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(args.seed))

        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=pin_mem)
        test_loader  = DataLoader(test_set,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=pin_mem)

    # -------------------------
    # model setup
    # -------------------------
    in_dim = dataset_train.X.shape[1] if (args.train_csv and args.test_csv) else dataset.X.shape[1]
    hidden = [int(x) for x in args.hidden.split(',') if x.strip()]
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=4, dropout=args.dropout).to(device)

    print(f"Dataset size: total={n_total}, train={n_train}, test={n_test}, input_dim={in_dim}")
    print("Training MLP:", model)
    print(f"Using device: {device}, pin_memory set to {pin_mem}")

    # train
    train_loop(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, ckpt_dir=args.ckpt_dir)

    # load best checkpoint if available
    best_ckpt = os.path.join(args.ckpt_dir, 'best_nointerp.pth')
    if os.path.exists(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck['model_state'])
        print(f"Loaded best checkpoint epoch={ck.get('epoch')} val_mae={ck.get('val_mae')}")

    # evaluate on test_set and save csvs
    preds, trues, dates = evaluate_model(model, test_loader, device)
    save_eval_csvs(preds, trues, dates, args.out_dir)
    print("Saved eval CSVs to", args.out_dir)

if __name__ == '__main__':
    main()