"""
Paired dataset for supervised downstream fine-tuning:
input = normalized single-cell Hi-C
label = normalized enhanced single-cell Hi-C
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_split_folder(data_path, split):
    """Use val/ as the canonical validation folder, with valid/ as a fallback."""
    if split in ("val", "valid"):
        for folder_name in ("val", "valid"):
            if os.path.isdir(os.path.join(data_path, folder_name)):
                return folder_name
        return "val"
    return split


class PairedNormalizedHiCDataset(Dataset):
    """
    Expected folder structure:
    data_path/
      train/
        <input_filename>.npy
        <label_filename>.npy
      val/ (valid/ is also accepted for backward compatibility)
        <input_filename>.npy
        <label_filename>.npy
      test/
        <input_filename>.npy
        <label_filename>.npy
    """

    def __init__(
        self,
        data_path,
        split="train",
        input_filename="normalized_distance_matrices.npy",
        label_filename="normalized_enhanced_distance_matrices.npy",
    ):
        super().__init__()
        self.data_path = data_path
        self.split = split

        folder_name = _resolve_split_folder(data_path, split)
        split_dir = os.path.join(data_path, folder_name)

        input_file = os.path.join(split_dir, input_filename)
        label_file = os.path.join(split_dir, label_filename)
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        self.inputs = np.load(input_file, mmap_mode="r")
        self.labels = np.load(label_file, mmap_mode="r")
        if self.inputs.shape != self.labels.shape:
            raise ValueError(
                f"Shape mismatch: inputs={self.inputs.shape}, labels={self.labels.shape}"
            )
        self.n_samples = len(self.inputs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = np.array(self.inputs[idx]).astype(np.float32)
        y = np.array(self.labels[idx]).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0)
        y_t = torch.from_numpy(y).unsqueeze(0)
        return x_t, y_t
