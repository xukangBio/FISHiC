import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.peft import apply_lora_to_model
from models.unet import SimpleUNet, UNet
from utils.paired_dataset import PairedNormalizedHiCDataset


def build_model(model_type: str, base_channels: int):
    if model_type == "unet":
        return UNet(
            in_channels=1,
            base_channels=base_channels,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=256,
            dropout=0.1,
            use_attention=True,
        )
    return SimpleUNet(
        in_channels=1,
        base_channels=base_channels,
        time_emb_dim=128,
    )


def symmetrize_contact_maps(x):
    x = np.asarray(x)
    if x.ndim != 4:
        return x
    h, w = x.shape[2], x.shape[3]
    if h != w:
        return x
    return ((x + np.transpose(x, (0, 1, 3, 2))) * 0.5).astype(x.dtype, copy=False)


def evaluate_regression_metrics(all_true, all_pred):
    true_v = all_true.reshape(all_true.shape[0], -1)
    pred_v = all_pred.reshape(all_pred.shape[0], -1)

    mse = float(np.mean((pred_v - true_v) ** 2))
    mae = float(np.mean(np.abs(pred_v - true_v)))
    pcc_list = []
    for i in range(true_v.shape[0]):
        a = true_v[i]
        b = pred_v[i]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        pcc_list.append(float(np.corrcoef(a, b)[0, 1]))
    pcc = float(np.mean(pcc_list)) if len(pcc_list) > 0 else float("nan")
    return {"mse_all": mse, "mae_all": mae, "pcc_all": pcc}


def forward_model(model, x):
    b = x.shape[0]
    t = torch.zeros(b, device=x.device, dtype=torch.float32)
    pred = model(x, t)
    return torch.clamp(pred, 0.0, 1.0)


def _preprocess_for_viz(mat, percentile):
    x = np.asarray(mat, dtype=np.float32)
    x = np.maximum(x, 0.0)
    x = np.log1p(x)

    p = float(percentile)
    if 0 < p <= 100:
        vmax = float(np.percentile(x, p))
        if np.isfinite(vmax) and vmax > 0:
            x = np.clip(x, 0.0, vmax)

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if np.isfinite(xmax - xmin) and (xmax - xmin) > 1e-12:
        x = (x - xmin) / (xmax - xmin)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x


def visualize_triplet(inp, pred, target, save_path, clip_percentile=90.0):
    inp_viz = _preprocess_for_viz(inp, clip_percentile)
    pred_viz = _preprocess_for_viz(pred, clip_percentile)
    target_viz = _preprocess_for_viz(target, clip_percentile)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(inp_viz, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Input (normalized)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_viz, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Prediction (enhanced)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(target_viz, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("Label (enhanced)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def load_finetuned_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_cfg = checkpoint.get("base_config", checkpoint.get("config", {}))
    ft_cfg = checkpoint.get("finetune_config", {})
    model_type = base_cfg.get("model_type", "unet")
    base_channels = int(base_cfg.get("base_channels", 64))
    model = build_model(model_type=model_type, base_channels=base_channels)

    use_lora = bool(ft_cfg) and (not bool(ft_cfg.get("no_lora", False)))
    if use_lora:
        target_modules = [
            x.strip()
            for x in str(ft_cfg.get("lora_targets", "")).split(",")
            if x.strip()
        ]
        apply_lora_to_model(
            model,
            rank=int(ft_cfg.get("lora_rank", 8)),
            alpha=float(ft_cfg.get("lora_alpha", 16.0)),
            dropout=float(ft_cfg.get("lora_dropout", 0.0)),
            target_modules=target_modules if len(target_modules) > 0 else None,
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, base_cfg, ft_cfg


def main():
    parser = argparse.ArgumentParser(description="Inference for normalized->enhanced downstream task")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_filename", type=str, default="normalized_distance_matrices.npy")
    parser.add_argument("--label_filename", type=str, default="normalized_enhanced_distance_matrices.npy")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_vis", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--viz_format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--viz_clip_percentile", type=float, default=90.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, base_cfg, ft_cfg = load_finetuned_model(args.checkpoint, args.device)
    if args.data_path is None:
        args.data_path = ft_cfg.get("data_path")
    if args.data_path is None:
        raise ValueError("data_path is required, or should be available in checkpoint finetune_config.")

    ds = PairedNormalizedHiCDataset(
        args.data_path,
        split=args.split,
        input_filename=args.input_filename,
        label_filename=args.label_filename,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Loaded {args.split} set: {len(ds)} samples")

    all_input = []
    all_pred = []
    all_true = []
    with torch.no_grad():
        for x_in, y in tqdm(loader, desc="Enhanced inference"):
            x_in = x_in.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            pred = forward_model(model, x_in)
            all_input.append(x_in.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())

    all_input = np.concatenate(all_input, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    all_input = symmetrize_contact_maps(all_input)
    all_pred = symmetrize_contact_maps(all_pred)
    all_true = symmetrize_contact_maps(all_true)

    metrics = evaluate_regression_metrics(all_true, all_pred)

    print(f"MSE(all): {metrics['mse_all']:.6f}")
    print(f"MAE(all): {metrics['mae_all']:.6f}")
    print(f"PCC(all): {metrics['pcc_all']:.4f}")

    np.save(os.path.join(args.output_dir, "input_matrices.npy"), all_input)
    np.save(os.path.join(args.output_dir, "pred_enhanced.npy"), all_pred)
    np.save(os.path.join(args.output_dir, "label_enhanced.npy"), all_true)

    with open(os.path.join(args.output_dir, "metrics_enhanced.json"), "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": vars(args),
                "base_config": base_cfg,
                "finetune_config": ft_cfg,
            },
            f,
            indent=2,
        )

    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for i in range(min(args.num_vis, len(all_true))):
        visualize_triplet(
            all_input[i, 0],
            all_pred[i, 0],
            all_true[i, 0],
            os.path.join(vis_dir, f"sample_{i:03d}_enhanced.{args.viz_format}"),
            clip_percentile=args.viz_clip_percentile,
        )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
