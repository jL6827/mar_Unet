#!/usr/bin/env python3
"""
Train on raw observation points (no spatial interpolation).
Uses dataset.RawPointDataset and utils.load_csv (which maps common column names).
Loss: MAE (L1). Split: 85%/15% by default (random).
Outputs eval_per_var.csv and eval_per_day.csv in --out_dir.
"""

# usage example:
# python train.py --csv processed_data_mean.csv --epochs 60 --batch 512 --lr 1e-3 --posenc --posenc_freqs 6 --val_frac 0.15 --hidden 256,256,128 --ckpt_dir checkpoints_nointerp --out_dir eval_nointerp

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
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
# Helpers: metrics & eval save
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
            # dates is list of ISO strings (or possibly date objects), extend directly
            all_dates.extend(dates)
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    return preds, trues, all_dates

def save_eval_csvs(preds, trues, dates, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # per-var MAE
    mae_per_var = np.mean(np.abs(preds - trues), axis=0)
    count = preds.shape[0]
    df_var = pd.DataFrame({
        'variable': ['so','thetao','uo','vo'],
        'mae': mae_per_var,
        'count': [count]*4
    })
    df_var.to_csv(os.path.join(out_dir, 'eval_per_var.csv'), index=False)

    # per-day aggregation
    day_map = defaultdict(list)
    for i, d in enumerate(dates):
        # d may be an ISO string or a date object. Use ISO string as key for safe sorting.
        if isinstance(d, str):
            key = d
        else:
            try:
                key = d.isoformat()
            except Exception:
                key = str(d)
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
# Training
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

        # val
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
    parser.add_argument('--csv', default='processed_data_mean.csv', help='input csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=str, default='256,256,128', help='comma-separated hidden sizes')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--posenc', action='store_true', help='use positional encodings')
    parser.add_argument('--posenc_freqs', type=int, default=6)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints_nointerp')
    parser.add_argument('--out_dir', type=str, default='eval_nointerp')
    args = parser.parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data (auto-maps column names)
    df = load_csv(args.csv)

    # create dataset using raw points (no interpolation)
    dataset = RawPointDataset(df, use_posenc=args.posenc, posenc_freqs=args.posenc_freqs, scale=True)
    n = len(dataset)
    n_test = max(1, int(n * args.val_frac))
    n_train = n - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(args.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    in_dim = dataset.X.shape[1]
    hidden = [int(x) for x in args.hidden.split(',') if x.strip()]
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=4, dropout=args.dropout).to(device)

    print(f"Dataset size: total={n}, train={n_train}, test={n_test}, input_dim={in_dim}")
    print("Training MLP:", model)

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