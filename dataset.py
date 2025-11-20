import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler

class RawPointDataset(Dataset):
    """
    Dataset using raw observation points (no interpolation).
    Expects DataFrame with canonical columns:
      time, lon, lat, depth, so, thetao, uo, vo

    Returns (x, y, date_str) where:
      x: torch.FloatTensor shape (D,) features per point
      y: torch.FloatTensor shape (4,) [so, thetao, uo, vo]
      date_str: ISO date string 'YYYY-MM-DD' (safe for DataLoader collate)
    Options:
      - use_posenc: positional encoding on lon,lat,depth,doy (sin/cos with freqs)
      - posenc_freqs: number of frequencies (k=1..posenc_freqs)
      - scale: whether to StandardScale input features
    """
    def __init__(self, df, use_posenc=True, posenc_freqs=6, scale=True):
        # make a copy to avoid modifying caller's DataFrame
        self.df = df.copy().reset_index(drop=True)

        # parse time and compute date & day-of-year
        self.df['time'] = pd.to_datetime(self.df['time'], errors='raise')
        self.df['date'] = self.df['time'].dt.date
        self.df['doy'] = self.df['time'].dt.dayofyear.astype(np.float32)

        self.use_posenc = bool(use_posenc)
        self.freqs = int(posenc_freqs)

        # base features: lon, lat, depth, doy_norm
        doy_norm = (self.df['doy'].values.astype(np.float32) / 365.0).reshape(-1, 1)
        base = np.stack([
            self.df['lon'].values.astype(np.float32),
            self.df['lat'].values.astype(np.float32),
            self.df['depth'].values.astype(np.float32),
            doy_norm.reshape(-1)
        ], axis=1)  # shape (N,4)

        if self.use_posenc:
            encs = []
            # apply posenc separately on each base dimension
            for i in range(base.shape[1]):
                v = base[:, i:i+1]  # (N,1)
                for k in range(1, self.freqs + 1):
                    encs.append(np.sin(2.0 * np.pi * k * v))
                    encs.append(np.cos(2.0 * np.pi * k * v))
            enc = np.concatenate(encs, axis=1)  # (N, 2 * freqs * 4)
            X = np.concatenate([base.astype(np.float32), enc.astype(np.float32)], axis=1)
        else:
            X = base.astype(np.float32)

        # targets in order [so, thetao, uo, vo]
        Y = np.stack([
            self.df['so'].values.astype(np.float32),
            self.df['thetao'].values.astype(np.float32),
            self.df['uo'].values.astype(np.float32),
            self.df['vo'].values.astype(np.float32)
        ], axis=1).astype(np.float32)

        # optional scaling of X (fit on all data)
        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        # return ISO date string so DataLoader collate can batch without errors
        date_obj = self.df.at[idx, 'date']
        # date_obj is datetime.date -> convert to ISO string
        date_str = date_obj.isoformat() if date_obj is not None else ''
        return x, y, date_str