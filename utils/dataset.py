import numpy as np
import torch
from torch.utils.data import Dataset
import os


def _resolve_split_folder(data_path, split):
    """Use val/ as the canonical validation folder, with valid/ as a fallback."""
    if split in ('val', 'valid'):
        for folder_name in ('val', 'valid'):
            if os.path.isdir(os.path.join(data_path, folder_name)):
                return folder_name
        return 'val'
    return split


class FISHDistanceMatrixDataset(Dataset):
    """
    Args:
        data_path: Path to folder containing train/val/test subfolders
        split: 'train', 'val', or 'test'
        seed: Random seed for reproducibility
    """
    def __init__(self, data_path, split='train', seed=42):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.seed = seed

        folder_name = _resolve_split_folder(data_path, split)

        # Load distance data
        try:
            data_file = os.path.join(data_path, folder_name, 'normalized_distance_matrices.npy')
        except:
            data_file = os.path.join(data_path, 'normalized_matrices.npy')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.data = np.load(data_file, mmap_mode='r')
        self.n_samples = len(self.data)

        # test 阶段加载 predefined mask (True = masked/missing, False = observed)
        self.predefined_masks = None
        # if split in ('val', 'test'):
        if split in ('test'):
            try:
                mask_file = os.path.join(data_path, folder_name, 'distance_mask.npy')
            except:
                mask_file = os.path.join(data_path, folder_name, 'adj_mask.npy')
            if not os.path.exists(mask_file):
                raise FileNotFoundError(f"Mask file not found: {mask_file}")
            self.predefined_masks = np.load(mask_file, mmap_mode='r')
            assert self.predefined_masks.shape == self.data.shape, \
                f"Mask shape {self.predefined_masks.shape} doesn't match data shape {self.data.shape}"

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            matrix_masked: Masked distance matrix (1, H, W)
            matrix: Original distance matrix (1, H, W)
            mask: Binary mask (1, H, W) - 1=observed, 0=masked/missing
        """
        matrix = np.array(self.data[idx]).astype(np.float32)

        if self.predefined_masks is not None:
            # test: 使用 predefined mask (True = masked/missing)
            # 转为 1=observed, 0=masked
            predefined_mask = np.array(self.predefined_masks[idx])
            mask = (~predefined_mask).astype(np.float32)
            matrix_masked = matrix * mask
        else:
            mask = np.ones_like(matrix, dtype=np.float32)
            matrix_masked = matrix.copy()

        # Convert to torch tensors
        matrix_tensor = torch.from_numpy(matrix).unsqueeze(0)  # (1, H, W)
        matrix_masked_tensor = torch.from_numpy(matrix_masked).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return matrix_masked_tensor, matrix_tensor, mask_tensor

    def get_raw_matrix(self, idx):
        """Get original matrix without any processing."""
        matrix = np.array(self.data[idx]).astype(np.float32)
        return matrix

    def get_raw_mask(self, idx):
        if self.predefined_masks is not None:
            return np.array(self.predefined_masks[idx])
        return None


class FISHDatasetWithFixedMask(Dataset):
    """
    Dataset with fixed masks for evaluation.
    Uses the predefined masks from distance_mask.npy.
    """
    def __init__(self, data_path, split='test', seed=42):
        super().__init__()
        self.data_path = data_path
        self.split = split

        folder_name = _resolve_split_folder(data_path, split)

        # Load distance data
        try:
            data_file = os.path.join(data_path, folder_name, 'normalized_distance_matrices.npy')
            self.data = np.load(data_file, mmap_mode='r')
        except:
            data_file = os.path.join(data_path, 'normalized_matrices.npy')
            self.data = np.load(data_file, mmap_mode='r')
        self.data = np.nan_to_num(self.data)
        self.n_samples = len(self.data)

        # Load predefined mask
        try:
            mask_file = os.path.join(data_path, folder_name, 'distance_mask.npy')
            self.predefined_masks = np.load(mask_file, mmap_mode='r')

        except:
            mask_file = os.path.join(data_path, 'adj_masks.npy')
            self.predefined_masks = ~np.load(mask_file, mmap_mode='r')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        matrix = np.array(self.data[idx]).astype(np.float32)

        # Get predefined mask (True = masked/missing)
        predefined_mask = np.array(self.predefined_masks[idx])
        mask = (~predefined_mask).astype(np.float32)  # 1=observed, 0=masked

        matrix_masked = matrix * mask

        matrix_tensor = torch.from_numpy(matrix).unsqueeze(0)
        matrix_masked_tensor = torch.from_numpy(matrix_masked).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return matrix_masked_tensor, matrix_tensor, mask_tensor


def load_fish_datasets(data_path, seed=42, use_fixed_mask=False, splits=('train', 'val', 'test')):
    """
    Load FISH datasets for specified splits. Only requested splits are read from disk.
    NO normalization applied (data already 0-1 normalized).
    FISHDistanceMatrixDataset: 仅 train 为完整图；test 使用 distance_mask.npy。

    Args:
        data_path: Path to folder containing train/val/test subfolders
        seed: Random seed
        use_fixed_mask: Whether to use fixed masks for evaluation
        splits: Tuple of splits to load, e.g. ('train',) or ('train', 'test'). Default ('train','val','test').

    Returns:
        One dataset per requested split, in order. e.g. splits=('train',) -> (train_ds,); splits=('train','val','test') -> (train_ds, val_ds, test_ds).
    """
    DatasetClass = FISHDatasetWithFixedMask if use_fixed_mask else FISHDistanceMatrixDataset

    split_to_key = {'train': 'train', 'val': 'val', 'test': 'test'}
    result = []
    for s in splits:
        result.append(DatasetClass(data_path, split=split_to_key[s], seed=seed))
    return tuple(result)
