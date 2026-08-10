import numpy as np
import torch


def generate_fixed_rate_bin_mask(
    coords,
    distance_matrices,
    corrupt_rate,
    seed,
    mask_value=0,
):
    """
    Mask bins at a fixed rate for coords and distance matrices.

    The same bins are masked in both inputs for each sample.
    Masked bins in coords are replaced with `mask_value`, while
    the corresponding rows and columns in distance matrices are masked.

    Supports NumPy arrays and PyTorch tensors.

    Args:
        coords: Coordinate array/tensor, or None.
        distance_matrices: Distance matrix array/tensor, or None.
        corrupt_rate: Fraction of bins to mask, in [0, 1].
        seed: Random seed, or None.
        mask_value: Value used for masking.

    Returns:
        masked_coords, masked_distance_matrices, coords_mask, distance_mask
    """
    if seed is not None:
        np.random.seed(seed)

    masked_coords = None
    masked_distance = None

    is_coords_torch = False
    is_dist_torch = False

    if coords is not None:
        is_coords_torch = isinstance(coords, torch.Tensor)
        masked_coords = coords.clone() if is_coords_torch else coords.copy()
        
    if distance_matrices is not None:
        is_dist_torch = isinstance(distance_matrices, torch.Tensor)
        masked_distance = distance_matrices.clone() if is_dist_torch else distance_matrices.copy()

    if masked_coords is None and masked_distance is None:
        return None, None, None, None

    def get_size_and_batch_for_coords(x):
        if x is None:
            return None, None
        shape = tuple(x.shape)
        if len(shape) == 1:  # (S,)
            return shape[0], 1
        if len(shape) == 2:
            if 3 in shape:  
                return (shape[0] if shape[1] == 3 else shape[1]), 1
            return shape[1], shape[0]
        if len(shape) == 3:
            if shape[2] == 3:
                return shape[1], shape[0]
            if shape[1] == 3:
                return shape[2], shape[0]
            return shape[1], shape[0]
        return None, None

    def get_size_and_batch_for_distance(x):
        if x is None:
            return None, None
        shape = tuple(x.shape)
        if len(shape) == 2 and shape[0] == shape[1]:  # (S,S)
            return shape[0], 1
        if len(shape) == 3:  # (N,S,S)
            return shape[1], shape[0]
        return None, None

    size_coords, batch_coords = get_size_and_batch_for_coords(masked_coords)
    size_dist, batch_dist = get_size_and_batch_for_distance(masked_distance)

    size = size_coords if size_coords is not None else size_dist
    if size is None:
        return masked_coords, masked_distance, None, None

    batch_c = batch_coords if batch_coords is not None else 1
    batch_d = batch_dist if batch_dist is not None else 1
    batch = max(batch_c, batch_d)

    def get_sample_coord(idx):
        if masked_coords is None:
            return None
        if is_coords_torch:
            if masked_coords.dim() == 1:
                return masked_coords
            if masked_coords.dim() == 2:
                return masked_coords
            return masked_coords[idx]
        else:
            if masked_coords.ndim == 1:
                return masked_coords
            if masked_coords.ndim == 2:
                return masked_coords
            return masked_coords[idx]

    def get_sample_dist(idx):
        if masked_distance is None:
            return None
        if is_dist_torch:
            if masked_distance.dim() == 2:
                return masked_distance
            return masked_distance[idx]
        else:
            if masked_distance.ndim == 2:
                return masked_distance
            return masked_distance[idx]

    def get_sample_coords_mask(idx):
        if coords_mask is None:
            return None
        if is_coords_torch:
            if coords_mask.dim() == 1:
                return coords_mask
            if coords_mask.dim() == 2:
                return coords_mask
            return coords_mask[idx]
        else:
            if coords_mask.ndim == 1:
                return coords_mask
            if coords_mask.ndim == 2:
                return coords_mask
            return coords_mask[idx]

    def get_sample_distance_mask(idx):
        if distance_mask is None:
            return None
        if is_dist_torch:
            if distance_mask.dim() == 2:
                return distance_mask
            return distance_mask[idx]
        else:
            if distance_mask.ndim == 2:
                return distance_mask
            return distance_mask[idx]

    corrupt_rate = float(corrupt_rate) if corrupt_rate is not None else 0.0
    corrupt_rate = max(0.0, min(1.0, corrupt_rate))
    num_bins_to_mask = int(round(size * corrupt_rate))
    if num_bins_to_mask <= 0:
        coords_mask = None
        distance_mask = None
        if masked_coords is not None:
            if is_coords_torch:
                coords_mask = torch.zeros_like(masked_coords, dtype=torch.bool)
            else:
                coords_mask = np.zeros_like(masked_coords, dtype=bool)
        if masked_distance is not None:
            if is_dist_torch:
                distance_mask = torch.zeros_like(masked_distance, dtype=torch.bool)
            else:
                distance_mask = np.zeros_like(masked_distance, dtype=bool)
        return masked_coords, masked_distance, coords_mask, distance_mask

    coords_mask = None
    distance_mask = None
    if masked_coords is not None:
        if is_coords_torch:
            coords_mask = torch.zeros_like(masked_coords, dtype=torch.bool)
        else:
            coords_mask = np.zeros_like(masked_coords, dtype=bool)
    if masked_distance is not None:
        if is_dist_torch:
            distance_mask = torch.zeros_like(masked_distance, dtype=torch.bool)
        else:
            distance_mask = np.zeros_like(masked_distance, dtype=bool)

    for i in range(batch):
        bins_to_mask = np.random.choice(size, num_bins_to_mask, replace=False)

        sample_c = get_sample_coord(i)
        sample_c_mask = get_sample_coords_mask(i)
        if sample_c is not None:
            if is_coords_torch:
                dim = sample_c.dim()
                if dim == 1:  # (S,)
                    sample_c[bins_to_mask] = mask_value
                    if sample_c_mask is not None:
                        sample_c_mask[bins_to_mask] = True
                elif dim == 2:
                    if sample_c.shape[1] == 3:  # (S,3)
                        sample_c[bins_to_mask, :] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[bins_to_mask, :] = True
                    elif sample_c.shape[0] == 3:  # (3,S)
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True
                    else: 
                        if sample_c.shape[1] == size:
                            sample_c[:, bins_to_mask] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[:, bins_to_mask] = True
                        else:
                            sample_c[bins_to_mask, :] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[bins_to_mask, :] = True
            else:
                dim = sample_c.ndim
                if dim == 1:  # (S,)
                    sample_c[bins_to_mask] = mask_value
                    if sample_c_mask is not None:
                        sample_c_mask[bins_to_mask] = True
                elif dim == 2:
                    if sample_c.shape[1] == 3:  # (S,3)
                        sample_c[bins_to_mask, :] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[bins_to_mask, :] = True
                    elif sample_c.shape[0] == 3:  # (3,S)
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True
                    else:
                        if sample_c.shape[1] == size:
                            sample_c[:, bins_to_mask] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[:, bins_to_mask] = True
                        else:
                            sample_c[bins_to_mask, :] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[bins_to_mask, :] = True
                elif dim == 3:
                    if sample_c.shape[-1] == 3:
                        sample_c[bins_to_mask, :] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[bins_to_mask, :] = True
                    elif sample_c.shape[-2] == 3:
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True
                    else:
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True

        sample_d = get_sample_dist(i)
        sample_d_mask = get_sample_distance_mask(i)
        if sample_d is not None:
            if is_dist_torch:
                sample_d[bins_to_mask, :] = mask_value
                sample_d[:, bins_to_mask] = mask_value
                if sample_d_mask is not None:
                    sample_d_mask[bins_to_mask, :] = True
                    sample_d_mask[:, bins_to_mask] = True
            else:
                sample_d[bins_to_mask, :] = mask_value
                sample_d[:, bins_to_mask] = mask_value
                if sample_d_mask is not None:
                    sample_d_mask[bins_to_mask, :] = True
                    sample_d_mask[:, bins_to_mask] = True

    return masked_coords, masked_distance, coords_mask, distance_mask

def generate_centered_bin_mask(
    coords,
    distance_matrices,
    center_positions,
    left_bin=1,
    right_bin=1,
    seed=None,
    mask_value=0,
):
    """
    Mask center bins and their neighboring bins.

    The same bins are masked in coords and distance matrices.

    Args:
        coords: Coordinate array/tensor, or None.
        distance_matrices: Distance matrix array/tensor, or None.
        center_positions: Center bin positions for each sample.
        left_bin: Number of bins masked to the left.
        right_bin: Number of bins masked to the right.
        seed: Random seed, or None.
        mask_value: Value used for masking.

    Returns:
        masked_coords, masked_distance_matrices, coords_mask, distance_mask
    """
    masked_coords = None
    masked_distance = None

    is_coords_torch = False
    is_dist_torch = False

    if coords is not None:
        is_coords_torch = isinstance(coords, torch.Tensor)
        masked_coords = coords.clone() if is_coords_torch else coords.copy()
        
    if distance_matrices is not None:
        is_dist_torch = isinstance(distance_matrices, torch.Tensor)
        masked_distance = distance_matrices.clone() if is_dist_torch else distance_matrices.copy()

    if masked_coords is None and masked_distance is None:
        return None, None, None, None

    def get_size_and_batch_for_coords(x):
        if x is None:
            return None, None
        shape = tuple(x.shape)
        if len(shape) == 1:  # (S,)
            return shape[0], 1
        if len(shape) == 2:
            if 3 in shape:
                return (shape[0] if shape[1] == 3 else shape[1]), 1
            return shape[1], shape[0]
        if len(shape) == 3:
            if shape[2] == 3:
                return shape[1], shape[0]
            if shape[1] == 3:
                return shape[2], shape[0]
            return shape[1], shape[0]
        return None, None

    def get_size_and_batch_for_distance(x):
        if x is None:
            return None, None
        shape = tuple(x.shape)
        if len(shape) == 2 and shape[0] == shape[1]:  # (S,S)
            return shape[0], 1
        if len(shape) == 3:  # (N,S,S)
            return shape[1], shape[0]
        return None, None

    size_coords, batch_coords = get_size_and_batch_for_coords(masked_coords)
    size_dist, batch_dist = get_size_and_batch_for_distance(masked_distance)

    size = size_coords if size_coords is not None else size_dist
    if size is None:
        return masked_coords, masked_distance, None, None

    batch_c = batch_coords if batch_coords is not None else 1
    batch_d = batch_dist if batch_dist is not None else 1
    batch = max(batch_c, batch_d)

    if isinstance(center_positions, (list, tuple, np.ndarray)):
        if len(center_positions) == 0:
            raise ValueError("center_positions cannot be empty")
        if isinstance(center_positions[0], (int, np.integer)):
            center_positions = [center_positions] * batch
        elif len(center_positions) != batch:
            raise ValueError(f"center_positions length ({len(center_positions)}) must match batch size ({batch})")
    else:
        raise ValueError("center_positions must be a list, tuple, or numpy array")

    def get_sample_coord(idx):
        if masked_coords is None:
            return None
        if is_coords_torch:
            if masked_coords.dim() == 1:
                return masked_coords
            if masked_coords.dim() == 2:
                return masked_coords
            return masked_coords[idx]
        else:
            if masked_coords.ndim == 1:
                return masked_coords
            if masked_coords.ndim == 2:
                return masked_coords
            return masked_coords[idx]

    def get_sample_dist(idx):
        if masked_distance is None:
            return None
        if is_dist_torch:
            if masked_distance.dim() == 2:
                return masked_distance
            return masked_distance[idx]
        else:
            if masked_distance.ndim == 2:
                return masked_distance
            return masked_distance[idx]

    coords_mask = None
    distance_mask = None
    if masked_coords is not None:
        if is_coords_torch:
            coords_mask = torch.zeros_like(masked_coords, dtype=torch.bool)
        else:
            coords_mask = np.zeros_like(masked_coords, dtype=bool)
    if masked_distance is not None:
        if is_dist_torch:
            distance_mask = torch.zeros_like(masked_distance, dtype=torch.bool)
        else:
            distance_mask = np.zeros_like(masked_distance, dtype=bool)

    for i in range(batch):
        sample_center_positions = center_positions[i]
        bins_to_mask = set()
        for center_pos in sample_center_positions:
            start = max(0, center_pos - left_bin)
            end = min(size, center_pos + right_bin + 1)
            for bin_idx in range(start, end):
                bins_to_mask.add(bin_idx)
        
        bins_to_mask = list(bins_to_mask)

        sample_c = get_sample_coord(i)
        if sample_c is not None:
            if is_coords_torch:
                dim = sample_c.dim()
                if dim == 1:  # (S,)
                    sample_c[bins_to_mask] = mask_value
                    if coords_mask is not None:
                        coords_mask[bins_to_mask] = True
                elif dim == 2:
                    if sample_c.shape[1] == 3:  # (S,3)
                        sample_c[bins_to_mask, :] = mask_value
                        if coords_mask is not None:
                            coords_mask[bins_to_mask, :] = True
                    elif sample_c.shape[0] == 3:  # (3,S)
                        sample_c[:, bins_to_mask] = mask_value
                        if coords_mask is not None:
                            coords_mask[:, bins_to_mask] = True
                    else:
                        if sample_c.shape[1] == size:
                            sample_c[:, bins_to_mask] = mask_value
                            if coords_mask is not None:
                                coords_mask[:, bins_to_mask] = True
                        else:
                            sample_c[bins_to_mask, :] = mask_value
                            if coords_mask is not None:
                                coords_mask[bins_to_mask, :] = True
            else:
                dim = sample_c.ndim
                if dim == 1:  # (S,)
                    sample_c[bins_to_mask] = mask_value
                    if coords_mask is not None:
                        coords_mask[bins_to_mask] = True
                elif dim == 2:
                    if sample_c.shape[1] == 3:  # (S,3)
                        sample_c[bins_to_mask, :] = mask_value
                        if coords_mask is not None:
                            coords_mask[bins_to_mask, :] = True
                    elif sample_c.shape[0] == 3:  # (3,S)
                        sample_c[:, bins_to_mask] = mask_value
                        if coords_mask is not None:
                            coords_mask[:, bins_to_mask] = True
                    else:
                        if sample_c.shape[1] == size:
                            sample_c[:, bins_to_mask] = mask_value
                            if coords_mask is not None:
                                coords_mask[:, bins_to_mask] = True
                        else:
                            sample_c[bins_to_mask, :] = mask_value
                            if coords_mask is not None:
                                coords_mask[bins_to_mask, :] = True

        sample_d = get_sample_dist(i)
        if sample_d is not None:
            if is_dist_torch:
                sample_d[bins_to_mask, :] = mask_value
                sample_d[:, bins_to_mask] = mask_value
                if distance_mask is not None:
                    distance_mask[bins_to_mask, :] = True
                    distance_mask[:, bins_to_mask] = True
            else:
                sample_d[bins_to_mask, :] = mask_value
                sample_d[:, bins_to_mask] = mask_value
                if distance_mask is not None:
                    distance_mask[bins_to_mask, :] = True
                    distance_mask[:, bins_to_mask] = True

    return masked_coords, masked_distance, coords_mask, distance_mask 


def generate_variable_bin_mask(
    coords,
    distance_matrices,
    bins,
    seed,
    mask_value=0,
):
    """
    Randomly mask a variable number of bins for each sample.

    The same bins are masked in coords and distance matrices.
    If `bins` is a range, the number of masked bins is sampled per sample.

    Args:
        coords: Coordinate array/tensor, or None.
        distance_matrices: Distance matrix array/tensor, or None.
        bins: Number of bins or a (min, max) range.
        seed: Base random seed, or None.
        mask_value: Value used for masking.

    Returns:
        masked_coords, masked_distance_matrices, coords_mask, distance_mask
    """
    masked_coords = None
    masked_distance = None

    is_coords_torch = False
    is_dist_torch = False

    if coords is not None:
        is_coords_torch = isinstance(coords, torch.Tensor)
        masked_coords = coords.clone() if is_coords_torch else coords.copy()
        
    if distance_matrices is not None:
        is_dist_torch = isinstance(distance_matrices, torch.Tensor)
        masked_distance = distance_matrices.clone() if is_dist_torch else distance_matrices.copy()

    if masked_coords is None and masked_distance is None:
        return None, None, None, None

    def get_size_and_batch_for_coords(x):
        if x is None:
            return None, None
        shape = tuple(x.shape)
        if len(shape) == 1:  # (S,)
            return shape[0], 1
        if len(shape) == 2:
            if 3 in shape:
                return (shape[0] if shape[1] == 3 else shape[1]), 1
            return shape[1], shape[0]
        if len(shape) == 3:
            if shape[2] == 3:
                return shape[1], shape[0]
            if shape[1] == 3:
                return shape[2], shape[0]
            return shape[1], shape[0]
        return None, None

    def get_size_and_batch_for_distance(x):
        if x is None:
            return None, None
        shape = tuple(x.shape)
        if len(shape) == 2 and shape[0] == shape[1]:  # (S,S)
            return shape[0], 1
        if len(shape) == 3:  # (N,S,S)
            return shape[1], shape[0]
        return None, None

    size_coords, batch_coords = get_size_and_batch_for_coords(masked_coords)
    size_dist, batch_dist = get_size_and_batch_for_distance(masked_distance)

    # 统一 size 与 batch
    size = size_coords if size_coords is not None else size_dist
    if size is None:
        return masked_coords, masked_distance, None, None

    batch_c = batch_coords if batch_coords is not None else 1
    batch_d = batch_dist if batch_dist is not None else 1
    batch = max(batch_c, batch_d)

    def get_sample_coord(idx):
        if masked_coords is None:
            return None
        if is_coords_torch:
            if masked_coords.dim() == 1:
                return masked_coords
            if masked_coords.dim() == 2:
                return masked_coords
            return masked_coords[idx]
        else:
            if masked_coords.ndim == 1:
                return masked_coords
            if masked_coords.ndim == 2:
                return masked_coords
            return masked_coords[idx]

    def get_sample_dist(idx):
        if masked_distance is None:
            return None
        if is_dist_torch:
            if masked_distance.dim() == 2:
                return masked_distance
            return masked_distance[idx]
        else:
            if masked_distance.ndim == 2:
                return masked_distance
            return masked_distance[idx]

    def get_sample_coords_mask(idx):
        if coords_mask is None:
            return None
        if is_coords_torch:
            if coords_mask.dim() == 1:
                return coords_mask
            if coords_mask.dim() == 2:
                return coords_mask
            return coords_mask[idx]
        else:
            if coords_mask.ndim == 1:
                return coords_mask
            if coords_mask.ndim == 2:
                return coords_mask
            return coords_mask[idx]

    def get_sample_distance_mask(idx):
        if distance_mask is None:
            return None
        if is_dist_torch:
            if distance_mask.dim() == 2:
                return distance_mask
            return distance_mask[idx]
        else:
            if distance_mask.ndim == 2:
                return distance_mask
            return distance_mask[idx]

    coords_mask = None
    distance_mask = None
    if masked_coords is not None:
        if is_coords_torch:
            coords_mask = torch.zeros_like(masked_coords, dtype=torch.bool)
        else:
            coords_mask = np.zeros_like(masked_coords, dtype=bool)
    if masked_distance is not None:
        if is_dist_torch:
            distance_mask = torch.zeros_like(masked_distance, dtype=torch.bool)
        else:
            distance_mask = np.zeros_like(masked_distance, dtype=bool)

    for i in range(batch):
        sample_seed = None if seed is None else seed + i
        rng = np.random.RandomState(sample_seed)
        
        if isinstance(bins, (tuple, list)):
            num_bins_to_mask = rng.randint(bins[0], bins[1] + 1)
        else:
            num_bins_to_mask = int(bins) if bins is not None else 0
        
        num_bins_to_mask = max(0, min(size, num_bins_to_mask))
        
        if num_bins_to_mask <= 0:
            continue
        
        bins_to_mask = rng.choice(size, num_bins_to_mask, replace=False)

        sample_c = get_sample_coord(i)
        sample_c_mask = get_sample_coords_mask(i)
        if sample_c is not None:
            if is_coords_torch:
                dim = sample_c.dim()
                if dim == 1:  # (S,)
                    sample_c[bins_to_mask] = mask_value
                    if sample_c_mask is not None:
                        sample_c_mask[bins_to_mask] = True
                elif dim == 2:
                    if sample_c.shape[1] == 3:  # (S,3)
                        sample_c[bins_to_mask, :] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[bins_to_mask, :] = True
                    elif sample_c.shape[0] == 3:  # (3,S)
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True
                    else:
                        if sample_c.shape[1] == size:
                            sample_c[:, bins_to_mask] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[:, bins_to_mask] = True
                        else:
                            sample_c[bins_to_mask, :] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[bins_to_mask, :] = True
            else:
                dim = sample_c.ndim
                if dim == 1:  # (S,)
                    sample_c[bins_to_mask] = mask_value
                    if sample_c_mask is not None:
                        sample_c_mask[bins_to_mask] = True
                elif dim == 2:
                    if sample_c.shape[1] == 3:  # (S,3)
                        sample_c[bins_to_mask, :] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[bins_to_mask, :] = True
                    elif sample_c.shape[0] == 3:  # (3,S)
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True
                    else:
                        if sample_c.shape[1] == size:
                            sample_c[:, bins_to_mask] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[:, bins_to_mask] = True
                        else:
                            sample_c[bins_to_mask, :] = mask_value
                            if sample_c_mask is not None:
                                sample_c_mask[bins_to_mask, :] = True
                elif dim == 3:
                    if sample_c.shape[-1] == 3:
                        sample_c[bins_to_mask, :] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[bins_to_mask, :] = True
                    elif sample_c.shape[-2] == 3:
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True
                    else:
                        sample_c[:, bins_to_mask] = mask_value
                        if sample_c_mask is not None:
                            sample_c_mask[:, bins_to_mask] = True

        sample_d = get_sample_dist(i)
        sample_d_mask = get_sample_distance_mask(i)
        if sample_d is not None:
            if is_dist_torch:
                sample_d[bins_to_mask, :] = mask_value
                sample_d[:, bins_to_mask] = mask_value
                if sample_d_mask is not None:
                    sample_d_mask[bins_to_mask, :] = True
                    sample_d_mask[:, bins_to_mask] = True
            else:
                sample_d[bins_to_mask, :] = mask_value
                sample_d[:, bins_to_mask] = mask_value
                if sample_d_mask is not None:
                    sample_d_mask[bins_to_mask, :] = True
                    sample_d_mask[:, bins_to_mask] = True

    return masked_coords, masked_distance, coords_mask, distance_mask