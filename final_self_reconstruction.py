import numpy as np
from sklearn.metrics import euclidean_distances
import numpy as np
from scipy.optimize import least_squares


def predict_missing_coordinates_enhanced(coords, dist_matrix, weight_power=0.3, n_iter=2):
    known_mask = ~np.isnan(coords).any(axis=1)
    unknown_mask = np.isnan(coords).any(axis=1)
    coords_complete = coords.copy()
    
    for i in np.where(unknown_mask)[0]:
        coords_complete[i] = np.random.rand(3) * np.nanmean(dist_matrix)
    
    for i in np.where(unknown_mask)[0]:
        known_dist_mask = ~np.isnan(dist_matrix[i]) & known_mask
        anchor_indices = np.where(known_dist_mask)[0]
        
        if len(anchor_indices) < 3:
            continue
            
        anchors = coords_complete[anchor_indices]
        dists = dist_matrix[i, anchor_indices]
        
        safe_dists = np.maximum(dists, 1e-6)
        weights = 1.0 / (safe_dists ** weight_power)
        weights /= np.sum(weights)
        
        def residuals(p):
            return weights * (np.linalg.norm(p - anchors, axis=1) - dists)
        
        centroid = np.average(anchors, axis=0, weights=weights)
        result = least_squares(residuals, centroid, method='lm')
        coords_complete[i] = result.x
    
    for _ in range(n_iter):
        optimize_mask = unknown_mask.copy()
        
        for i in np.where(unknown_mask)[0]:
            connected_points = ~np.isnan(dist_matrix[i])
            optimize_mask = optimize_mask | (connected_points & known_mask)
        
        for i in np.where(optimize_mask)[0]:
            valid_dists_mask = ~np.isnan(dist_matrix[i])
            neighbor_indices = np.where(valid_dists_mask)[0]
            
            if len(neighbor_indices) < 3:
                continue
                
            neighbors = coords_complete[neighbor_indices]
            dists = dist_matrix[i, neighbor_indices]
            
            safe_dists = np.maximum(dists, 1e-6)
            weights = np.where(known_mask[neighbor_indices], 
                             1.0/(safe_dists**weight_power), 
                             0.5/(safe_dists**weight_power))
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights /= np.sum(weights) + 1e-10
            
            def joint_residuals(p):
                diff = np.linalg.norm(p - neighbors, axis=1) - dists
                return np.nan_to_num(weights * diff, nan=0.0)
            
            try:
                result = least_squares(joint_residuals, coords_complete[i], method='lm')
                coords_complete[i] = result.x
            except:
                continue
    
    if np.sum(known_mask) >= 3:
        known_points = coords[known_mask]
        predicted_points = coords_complete[known_mask]
        
        centroid_known = np.mean(known_points, axis=0)
        centroid_pred = np.mean(predicted_points, axis=0)
        H = (predicted_points - centroid_pred).T @ (known_points - centroid_known)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        T = centroid_known - R @ centroid_pred
        
        coords_complete[unknown_mask] = (coords_complete[unknown_mask] @ R) + T
    
    return coords_complete