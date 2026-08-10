"""
FISH Data Preprocessing Pipeline for Pre-train Dataset

Steps:
1. Read Use_FISH.txt to get Pre-train 4DN IDs
2. Download FISH data CSV files from 4DN database
3. Filter cells with missing rate < 10%
4. Interpolate missing XYZ coordinates
5. Convert to distance matrices
6. Extract 64x64 diagonal patches
7. Split into train/test/val (8:1:1)
8. Save as npy files

Usage (on Windows with data at H:\FISH_data):
    python preprocess_fish_data.py --input-dir H:\FISH_data --output-dir H:\FISH_data\preprocessed --patch-size 64
"""

import os
import re
import sys
import csv
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import requests


def parse_fish_csv(file_path):
    """
    Parse a 4DN FISH CSV file.

    Returns DataFrame with columns including X, Y, Z, Trace_ID, Spot_Index.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    columns = None
    data_rows = []

    for line in lines:
        line = line.strip()
        # Remove surrounding quotes
        if line.startswith('"'):
            end_quote = line.rfind('"')
            if end_quote > 0:
                line = line[1:end_quote]

        if not line:
            continue

        # Metadata lines
        if line.startswith('##'):
            if line.startswith('##columns='):
                col_match = re.search(r'\(([^)]+)\)', line)
                if col_match:
                    columns = [c.strip() for c in col_match.group(1).split(',') if c.strip()]
            continue

        if line.startswith('#'):
            continue

        # Data line
        if columns is not None:
            values = [v.strip() for v in line.split(',')]
            while values and values[-1] == '':
                values.pop()
            if len(values) == len(columns):
                data_rows.append(values)

    if columns is None or not data_rows:
        return None

    df = pd.DataFrame(data_rows, columns=columns)
    return df


def find_trace_id_column(df):
    """Find the Trace_ID column name (case-insensitive)."""
    for col in df.columns:
        if col.strip().upper() == 'TRACE_ID':
            return col
    return None


def find_xyz_columns(df):
    """Find X, Y, Z coordinate columns."""
    x_col, y_col, z_col = None, None, None
    for col in df.columns:
        col_upper = col.strip().upper()
        if col_upper == 'X':
            x_col = col
        elif col_upper == 'Y':
            y_col = col
        elif col_upper == 'Z':
            z_col = col
    return x_col, y_col, z_col


def compute_cell_missing_rate(cell_df, expected_spots, spot_index_col=None):
    """
    Compute missing rate for a single cell.

    Missing rate = (expected_spots - actual_spots) / expected_spots
    where actual_spots is the number of unique spot indices for this cell.
    """
    if spot_index_col and spot_index_col in cell_df.columns:
        actual_spots = cell_df[spot_index_col].nunique()
    else:
        actual_spots = len(cell_df)
    return (expected_spots - actual_spots) / expected_spots if expected_spots > 0 else 0


def interpolate_cell_coordinates(cell_df, x_col, y_col, z_col, spot_index_col, expected_spots):
    """
    Interpolate missing XYZ coordinates for a cell using linear interpolation.

    The idea: for each locus index from 0 to expected_spots-1, if no spot exists,
    interpolate from neighboring spots.
    """
    # Get existing spots
    existing = []
    for _, row in cell_df.iterrows():
        try:
            idx = int(row[spot_index_col])
            x = float(row[x_col])
            y = float(row[y_col])
            z = float(row[z_col])
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                existing.append((idx, x, y, z))
        except (ValueError, TypeError):
            continue

    if len(existing) < 2:
        return None  # Not enough points to interpolate

    existing.sort(key=lambda p: p[0])
    indices = [p[0] for p in existing]
    xs = [p[1] for p in existing]
    ys = [p[2] for p in existing]
    zs = [p[3] for p in existing]

    # Create interpolation functions (linear, allow extrapolation for boundary points)
    try:
        fx = interp1d(indices, xs, kind='linear', fill_value='extrapolate')
        fy = interp1d(indices, ys, kind='linear', fill_value='extrapolate')
        fz = interp1d(indices, zs, kind='linear', fill_value='extrapolate')
    except Exception:
        return None

    # Interpolate all loci from 0 to expected_spots-1
    all_indices = np.arange(expected_spots)
    interp_x = fx(all_indices)
    interp_y = fy(all_indices)
    interp_z = fz(all_indices)

    # Build full coordinate array
    coords = np.zeros((expected_spots, 3))
    coords[:, 0] = interp_x
    coords[:, 1] = interp_y
    coords[:, 2] = interp_z

    return coords


def coords_to_distance_matrix(coords):
    """
    Convert XYZ coordinates to pairwise Euclidean distance matrix.

    Args:
        coords: (N, 3) array of XYZ coordinates

    Returns:
        dist_matrix: (N, N) symmetric distance matrix
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    return dist


def normalize_distance_matrix(dist_matrix):
    """
    Normalize distance matrix to [0, 1] range.
    """
    # Set diagonal to 0
    np.fill_diagonal(dist_matrix, 0)

    max_val = dist_matrix.max()
    if max_val > 0:
        dist_matrix = dist_matrix / max_val

    return dist_matrix


def extract_diagonal_patches(dist_matrix, patch_size=64):
    """
    Extract 64x64 patches along the diagonal of the distance matrix.

    Uses a sliding window approach with stride=patch_size//2 to get
    overlapping patches that cover the full matrix.

    Args:
        dist_matrix: (N, N) distance matrix
        patch_size: size of each patch (default 64)

    Returns:
        patches: list of (patch_size, patch_size) arrays
    """
    n = dist_matrix.shape[0]
    if n < patch_size:
        return []

    patches = []
    stride = patch_size // 2  # 50% overlap

    for start in range(0, n - patch_size + 1, stride):
        patch = dist_matrix[start:start + patch_size, start:start + patch_size]
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)

    return patches


def download_4dn_file(fourdn_id, output_dir, cache=True):
    """
    Download FISH data CSV file from 4DN Nucleome Database.

    The file URL pattern:
    https://data.4dnucleome.org/files-processed/4DNFIxxxx/@@download/4DNFIxxxx.csv
    """
    csv_path = os.path.join(output_dir, f'{fourdn_id}.csv')

    if cache and os.path.exists(csv_path):
        return csv_path

    # Try different URL patterns used by 4DN
    urls = [
        f'https://data.4dnucleome.org/files-processed/{fourdn_id}/@@download/{fourdn_id}.csv',
        f'https://data.4dnucleome.org/files/{fourdn_id}/@@download/{fourdn_id}.csv',
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(csv_path, 'wb') as f:
                    f.write(resp.content)
                return csv_path
        except Exception:
            continue

    return None


def get_pretrain_ids(use_fish_path):
    """
    Read Use_FISH.txt and return list of Pre-train 4DN IDs.
    """
    pretrain_ids = []
    seen = set()

    with open(use_fish_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                fourdn_id = parts[0].strip()
                label = parts[1].strip()
                if label == 'Pre-train' and fourdn_id not in seen:
                    seen.add(fourdn_id)
                    pretrain_ids.append(fourdn_id)

    return pretrain_ids


def find_spot_index_column(df):
    """Find the Spot_Index column."""
    for col in df.columns:
        if col.strip().upper() in ('SPOT_INDEX', 'LOCUS_INDEX', 'BIN_INDEX'):
            return col
    return None


def process_single_4dn(fourdn_id, csv_path, patch_size=64, missing_rate_threshold=0.10):
    """
    Process a single 4DN FISH file.

    Returns list of 64x64 distance matrix patches.
    """
    df = parse_fish_csv(csv_path)
    if df is None:
        return [], 0, 0

    trace_col = find_trace_id_column(df)
    x_col, y_col, z_col = find_xyz_columns(df)
    spot_idx_col = find_spot_index_column(df)

    if trace_col is None or x_col is None or y_col is None:
        return [], 0, 0

    # Find expected spots (maximum spots per cell)
    if spot_idx_col:
        expected_spots = df[spot_idx_col].nunique()
    else:
        expected_spots = df[trace_col].value_counts().max()

    if expected_spots < patch_size:
        return [], 0, 0

    all_patches = []
    total_cells = 0
    filtered_cells = 0

    for trace_id, cell_df in df.groupby(trace_col):
        total_cells += 1

        # Compute missing rate
        if spot_idx_col:
            actual_unique = cell_df[spot_idx_col].nunique()
        else:
            actual_unique = len(cell_df)

        missing_rate = (expected_spots - actual_unique) / expected_spots if expected_spots > 0 else 0

        # Filter by missing rate
        if missing_rate > missing_rate_threshold:
            continue

        filtered_cells += 1

        # Interpolate coordinates
        if spot_idx_col:
            coords = interpolate_cell_coordinates(cell_df, x_col, y_col, z_col, spot_idx_col, expected_spots)
        else:
            # Without spot index, use row order
            coords = cell_df[[x_col, y_col, z_col]].values.astype(float)
            if len(coords) != expected_spots:
                continue
            coords = coords

        if coords is None:
            continue

        # Compute distance matrix
        dist_matrix = coords_to_distance_matrix(coords)
        dist_matrix = normalize_distance_matrix(dist_matrix)

        # Extract diagonal patches
        patches = extract_diagonal_patches(dist_matrix, patch_size)
        all_patches.extend(patches)

    return all_patches, total_cells, filtered_cells


def main():
    parser = argparse.ArgumentParser(description='Preprocess FISH data for pre-training')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing FISH CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--use-fish-file', type=str, default=None,
                       help='Path to Use_FISH.txt (default: input-dir/Use_FISH.txt)')
    parser.add_argument('--patch-size', type=int, default=64,
                       help='Patch size (default: 64)')
    parser.add_argument('--missing-rate', type=float, default=0.10,
                       help='Maximum missing rate threshold (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting (default: 42)')
    parser.add_argument('--download', action='store_true',
                       help='Download missing CSV files from 4DN database')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    patch_size = args.patch_size
    missing_rate_threshold = args.missing_rate

    # Find Use_FISH.txt
    if args.use_fish_file:
        use_fish_path = args.use_fish_file
    else:
        use_fish_path = os.path.join(input_dir, 'Use_FISH.txt')

    if not os.path.exists(use_fish_path):
        print(f"Error: Use_FISH.txt not found at {use_fish_path}")
        sys.exit(1)

    # Get Pre-train 4DN IDs
    pretrain_ids = get_pretrain_ids(use_fish_path)
    print(f"Found {len(pretrain_ids)} Pre-train 4DN IDs")

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Statistics
    all_patches = []
    stats = {}
    total_cells_all = 0
    total_cells_filtered = 0

    # Process each 4DN ID
    print(f"\nProcessing {len(pretrain_ids)} Pre-train experiments...")
    for fourdn_id in tqdm(pretrain_ids, desc='Experiments'):
        csv_path = os.path.join(input_dir, f'{fourdn_id}.csv')

        if not os.path.exists(csv_path):
            if args.download:
                csv_path = download_4dn_file(fourdn_id, input_dir)
                if csv_path is None:
                    stats[fourdn_id] = {'status': 'missing_csv'}
                    continue
            else:
                stats[fourdn_id] = {'status': 'missing_csv'}
                continue

        patches, total_cells, filtered_cells = process_single_4dn(
            fourdn_id, csv_path, patch_size, missing_rate_threshold
        )

        total_cells_all += total_cells
        total_cells_filtered += filtered_cells

        stats[fourdn_id] = {
            'status': 'ok',
            'total_cells': total_cells,
            'filtered_cells': filtered_cells,
            'num_patches': len(patches),
        }

        all_patches.extend(patches)

    # Convert to numpy array
    if len(all_patches) == 0:
        print("Error: No patches generated!")
        sys.exit(1)

    all_patches = np.array(all_patches, dtype=np.float32)
    print(f"\nTotal patches generated: {len(all_patches)}")
    print(f"Patch shape: {all_patches.shape}")

    # Split into train/test/val (8:1:1)
    np.random.seed(args.seed)
    n = len(all_patches)
    indices = np.random.permutation(n)

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = all_patches[train_indices]
    val_data = all_patches[val_indices]
    test_data = all_patches[test_indices]

    # Save npy files
    train_path = os.path.join(train_dir, 'train_data.npy')
    val_path = os.path.join(val_dir, 'val_data.npy')
    test_path = os.path.join(test_dir, 'test_data.npy')

    np.save(train_path, train_data)
    np.save(val_path, val_data)
    np.save(test_path, test_data)

    print(f"\nData split:")
    print(f"  Train: {train_data.shape} -> {train_path}")
    print(f"  Val:   {val_data.shape} -> {val_path}")
    print(f"  Test:  {test_data.shape} -> {test_path}")

    # Save statistics
    stats_summary = {
        'total_4dn_ids': len(pretrain_ids),
        'total_cells_processed': total_cells_all,
        'total_cells_after_filter': total_cells_filtered,
        'total_patches': len(all_patches),
        'train_patches': len(train_data),
        'val_patches': len(val_data),
        'test_patches': len(test_data),
        'patch_size': patch_size,
        'missing_rate_threshold': missing_rate_threshold,
        'per_experiment': stats,
    }

    stats_path = os.path.join(output_dir, 'processing_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\nStatistics saved to {stats_path}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
