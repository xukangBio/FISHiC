import numpy as np
import torch
from scipy.stats import pearsonr


def compute_insulation_score(matrix, window_size=5):
    n = matrix.shape[0]
    insulation_scores = np.zeros(n)

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size)

        if i > 0 and i < n - 1:
            upper = matrix[start:i, i:end]
            lower = matrix[i:end, start:i]

            insulation = (np.mean(upper) + np.mean(lower)) / 2
            insulation_scores[i] = insulation
        else:
            insulation_scores[i] = np.nan

    return insulation_scores


def compute_insulation_pcc(true_matrix, pred_matrix, mask, window_size=5):
    if true_matrix.ndim == 3:
        true_matrix = true_matrix.squeeze(0)
    if pred_matrix.ndim == 3:
        pred_matrix = pred_matrix.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    row_sums = mask.sum(axis=1)
    col_sums = mask.sum(axis=0)
    mask_1d = ((row_sums < mask.shape[1]) | (col_sums < mask.shape[0])).astype(bool)

    true_ins = compute_insulation_score(true_matrix, window_size)
    pred_ins = compute_insulation_score(pred_matrix, window_size)

    true_ins_imputed = true_ins[mask_1d]
    pred_ins_imputed = pred_ins[mask_1d]

    valid = ~(np.isnan(true_ins_imputed) | np.isnan(pred_ins_imputed))

    if valid.sum() < 2:
        return np.nan

    pcc, _ = pearsonr(true_ins_imputed[valid], pred_ins_imputed[valid])
    return pcc


def compute_pcc_at_imputed(true_matrix, pred_matrix, mask):
    if true_matrix.ndim == 3:
        true_matrix = true_matrix.squeeze(0)
    if pred_matrix.ndim == 3:
        pred_matrix = pred_matrix.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    imputed_mask = (mask == 0)

    true_flat = true_matrix[imputed_mask]
    pred_flat = pred_matrix[imputed_mask]

    if len(true_flat) < 2:
        return np.nan

    pcc, _ = pearsonr(true_flat, pred_flat)
    return pcc


def compute_mse_at_imputed(true_matrix, pred_matrix, mask):
    if true_matrix.ndim == 3:
        true_matrix = true_matrix.squeeze(0)
    if pred_matrix.ndim == 3:
        pred_matrix = pred_matrix.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    imputed_mask = (mask == 0)
    diff = (true_matrix - pred_matrix) ** 2
    mse = diff[imputed_mask].mean()
    return mse


def compute_mae_at_imputed(true_matrix, pred_matrix, mask):
    if true_matrix.ndim == 3:
        true_matrix = true_matrix.squeeze(0)
    if pred_matrix.ndim == 3:
        pred_matrix = pred_matrix.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    imputed_mask = (mask == 0)
    diff = np.abs(true_matrix - pred_matrix)
    mae = diff[imputed_mask].mean()
    return mae


def evaluate_imputation(true_matrix, pred_matrix, mask, window_size=5):
    metrics = {}

    metrics['pcc_imputed'] = compute_pcc_at_imputed(true_matrix, pred_matrix, mask)

    metrics['insulation_pcc'] = compute_insulation_pcc(true_matrix, pred_matrix, mask, window_size)

    metrics['mse_imputed'] = compute_mse_at_imputed(true_matrix, pred_matrix, mask)
    metrics['mae_imputed'] = compute_mae_at_imputed(true_matrix, pred_matrix, mask)

    mask_bool = (mask > 0)
    true_flat = true_matrix[mask_bool]
    pred_flat = pred_matrix[mask_bool]
    if len(true_flat) > 1:
        metrics['pcc_observed'], _ = pearsonr(true_flat, pred_flat)
    else:
        metrics['pcc_observed'] = np.nan

    return metrics


def batch_evaluate(true_matrices, pred_matrices, masks, window_size=5):
    all_metrics = {
        'pcc_imputed': [],
        'insulation_pcc': [],
        'mse_imputed': [],
        'mae_imputed': [],
    }

    N = true_matrices.shape[0]
    for i in range(N):
        metrics = evaluate_imputation(
            true_matrices[i], pred_matrices[i], masks[i], window_size
        )
        for key in all_metrics:
            if not np.isnan(metrics[key]):
                all_metrics[key].append(metrics[key])

    result = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            result[key] = np.mean(values)
            result[key + '_std'] = np.std(values)
        else:
            result[key] = np.nan
            result[key + '_std'] = np.nan

    return result
