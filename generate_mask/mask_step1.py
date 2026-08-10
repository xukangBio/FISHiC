import numpy as np
from mask_generator import generate_fixed_rate_bin_mask, generate_variable_bin_mask
from tqdm import tqdm
import os


if __name__ == "__main__":

    output_path = './masked_data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_path = '/home/xukang/model_92/InpaintTools/data_collection/2018_science/processed_data/HCT116_34-37'
    coords = np.load(os.path.join(data_path, 'coordinates_interpolated.npy'))
    distance = np.load(os.path.join(data_path, 'distance_matrices.npy'))
    distance_normalized = np.load(os.path.join(data_path, 'normalized_distance_matrices.npy'))
    sigma = np.load(os.path.join(data_path, 'sigma_values.npy'))
    
    seed = 100

    all_masked_coords = []
    all_masked_distance = []
    all_coords_mask = []
    all_distance_mask = []
    all_normalized_distance_matrices = []
    for i in tqdm(range(coords.shape[0])):
        ### True 表示被遮蔽（mask）的位置
        masked_coords, masked_distance, coords_mask, distance_mask = generate_fixed_rate_bin_mask(
            coords[i], distance[i], corrupt_rate=0.5, seed=seed + i, mask_value=None
        )
        # masked_coords, masked_distance, coords_mask, distance_mask = generate_variable_bin_mask(
        #     coords[i], distance[i], bins=(10,36), seed=seed + i, mask_value=None
        # )
        
        all_masked_coords.append(masked_coords)
        all_masked_distance.append(masked_distance)
        all_coords_mask.append(coords_mask)
        all_distance_mask.append(distance_mask)

    all_masked_coords = np.array(all_masked_coords)
    all_masked_distance = np.array(all_masked_distance)
    all_coords_mask = np.array(all_coords_mask)
    all_distance_mask = np.array(all_distance_mask)
    all_sigma = sigma

    np.save(os.path.join(output_path, 'masked_coords.npy'), all_masked_coords)
    np.save(os.path.join(output_path, 'masked_distance.npy'), all_masked_distance)
    np.save(os.path.join(output_path, 'coords_mask.npy'), all_coords_mask)
    np.save(os.path.join(output_path, 'distance_mask.npy'), all_distance_mask)
    np.save(os.path.join(output_path, 'normalized_distance_matrices.npy'), distance_normalized)
    np.save(os.path.join(output_path, 'sigma.npy'), all_sigma)
   
    np.save(os.path.join(output_path, 'ground_coords.npy'), coords)
    np.save(os.path.join(output_path, 'ground_distance.npy'), distance)
    np.save(os.path.join(output_path, 'ground_distance_normalized.npy'), distance_normalized)
    
