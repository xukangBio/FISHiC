import multiprocessing as mp
import warnings
import numpy as np
import os
from tqdm import tqdm
from final_self_reconstruction import predict_missing_coordinates_enhanced
from scipy.spatial import distance_matrix


warnings.filterwarnings('ignore')
def process_single(args):
    i, distance, coords_masked = args
    try:
        rec = predict_missing_coordinates_enhanced(coords_masked[i], distance[i])
        return rec
    except Exception as e:
        print(f"处理索引 {i} 时出错: {e}")
        return (None, None, None)
    

def main():
    res_distance_matrix = np.load('pred_finetuned.npy')
    sigma = np.load('test/sigma.npy')
    coordinates_masked = np.load('test/masked_coords.npy')


    raw_contact = np.squeeze(res_distance_matrix, axis=1)

    raw_contact = raw_contact
    raw_distance = np.zeros_like(raw_contact)

    for i in range(raw_contact.shape[0]):
        raw_contact[i] = (raw_contact[i] + raw_contact[i].T) / 2
        raw_contact[i] = np.clip(raw_contact[i], 0, 1) 
        raw_distance[i] = sigma[i] * np.sqrt(-np.log((raw_contact[i])))


    raw_distance = np.nan_to_num(raw_distance, posinf=0)

    os.makedirs('res', exist_ok=True)
    num_processes = min(mp.cpu_count(), 30) 
    total_samples = len(raw_distance)
    
    print(f"使用 {num_processes} 个进程处理 {total_samples} 个样本")

    params = [(i, raw_distance, coordinates_masked) 
                for i in range(total_samples)]

    with mp.Pool(processes=num_processes) as pool:
        rencon_coord = list(tqdm(pool.imap(process_single, params),
                            total=total_samples,
                            desc="处理进度"))


    rencon_coord = np.array(rencon_coord)

    distance_matrix_all = np.zeros((rencon_coord.shape[0], rencon_coord.shape[1], rencon_coord.shape[1]))
    for i in range(rencon_coord.shape[0]):
        distance_matrix_all[i] = distance_matrix(rencon_coord[i], rencon_coord[i])

    np.save('recon_coord.npy', rencon_coord)
    np.save('recon_distance_matrix.npy', distance_matrix_all)

if __name__ == "__main__":
    main()