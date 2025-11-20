import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from datetime import datetime

def load_csv(path):
    """
    Robust loader: accepts common alternative column names and renames them in-memory
    to canonical names expected by the training code:
      time, lon, lat, depth, uo, vo, so, thetao

    This function DOES NOT modify the CSV file on disk; it only returns a DataFrame
    with columns renamed if necessary.
    """
    df = pd.read_csv(path)

    # synonyms mapping: canonical -> aliases
    synonyms = {
        'lon': ['lon', 'longitude', 'long', 'lon_deg', 'longitude_deg', 'x'],
        'lat': ['lat', 'latitude', 'lat_deg', 'latitude_deg', 'y'],
        'time': ['time', 'timestamp', 'date', 'datetime', 'time_utc'],
        'depth': ['depth', 'Depth', 'depth_m', 'z'],
        'uo': ['uo', 'u_o', 'u', 'u_velocity'],
        'vo': ['vo', 'v_o', 'v', 'v_velocity'],
        'so': ['so', 'salinity', 'sal', 'salinity_psu'],
        'thetao': ['thetao', 'theta', 'temp', 'temperature', 'theta_o']
    }

    inv = {}
    for canon, alts in synonyms.items():
        for a in alts:
            inv[a.lower()] = canon

    rename_map = {}
    for c in df.columns:
        key = c.lower()
        if key in inv:
            # only rename if different
            if inv[key] != c:
                rename_map[c] = inv[key]

    if rename_map:
        print("In-memory column renames (applied):")
        for old, new in rename_map.items():
            print(f"  {old} -> {new}")
        df = df.rename(columns=rename_map)
    else:
        print("No column renaming needed.")

    required = {'time', 'lon', 'lat', 'depth', 'uo', 'vo', 'so', 'thetao'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}. Missing: {missing}")

    # parse time to datetime
    df['time'] = pd.to_datetime(df['time'], errors='raise')

    return df

# retained functions (not used in no-interp workflow) kept for backward compatibility
def make_grid(lon_min, lon_max, lat_min, lat_max, nx, ny):
    xs = np.linspace(lon_min, lon_max, nx)
    ys = np.linspace(lat_min, lat_max, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y

def interpolate_to_grid(points_lon, points_lat, values, grid_x, grid_y, method='linear'):
    pts = np.column_stack((points_lon, points_lat))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    try:
        grid_z = griddata(pts, values, grid_pts, method=method)
        if np.any(np.isnan(grid_z)):
            grid_z_near = griddata(pts, values, grid_pts, method='nearest')
            nan_mask = np.isnan(grid_z)
            grid_z[nan_mask] = grid_z_near[nan_mask]
    except Exception:
        grid_z = griddata(pts, values, grid_pts, method='nearest')
    return grid_z.reshape(grid_x.shape)

def depth_binning(depths, n_bins):
    dmin, dmax = np.nanmin(depths), np.nanmax(depths)
    bins = np.linspace(dmin, dmax, n_bins+1)
    centers = 0.5*(bins[:-1]+bins[1:])
    return bins, centers