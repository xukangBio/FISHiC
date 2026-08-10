import numpy as np
import os


def split_indices(n, ratios=(0.7, 0.1, 0.2), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]
    return train_idx, valid_idx, test_idx


def save_split(arr_dict, base_dir, split_name, indices):
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for fname, arr in arr_dict.items():
        if arr is None:  # 兼容可能不存在的文件
            continue
        np.save(os.path.join(split_dir, fname), arr[indices])


def split_dataset_for_dir(base_dir, dest_dir=None, seed=42):
    files = {
        'masked_coords.npy': None,
        'masked_distance.npy': None,
        'coords_mask.npy': None,
        'distance_mask.npy': None,
        'normalized_distance_matrices.npy': None,
        'sigma.npy': None,
        'ground_coords.npy': None,
        'ground_distance.npy': None,
        'ground_distance_normalized.npy': None,
        
        
    }
    for k in list(files.keys()):
        path = os.path.join(base_dir, k)
        if os.path.exists(path):
            files[k] = np.load(path, allow_pickle=True)
    if files['masked_coords.npy'] is None and files['masked_distance.npy'] is None:
        print(f"跳过 {base_dir} （未找到主数据文件）")
        return
    ref = files['masked_coords.npy']
    if ref is None:
        ref = files['masked_distance.npy']
    n = ref.shape[0]
    train_idx, valid_idx, test_idx = split_indices(n, (0.7, 0.1, 0.2), seed=seed)
    if dest_dir is None:
        dest_dir = base_dir
    os.makedirs(dest_dir, exist_ok=True)
    print(f"{base_dir} -> {dest_dir}: N={n} -> train {len(train_idx)}, val {len(valid_idx)}, test {len(test_idx)}")

    save_split(files, dest_dir, 'train', train_idx)
    save_split(files, dest_dir, 'val', valid_idx)
    save_split(files, dest_dir, 'test', test_idx)


if __name__ == "__main__":
    source_root = 'masked_data'
    dest_root = 'masked_data_Dataset_split'
    os.makedirs(source_root, exist_ok=True)
    os.makedirs(dest_root, exist_ok=True)

    split_dataset_for_dir(source_root, dest_root, seed=100)

        
