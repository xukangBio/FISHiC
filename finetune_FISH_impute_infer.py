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

from models.fish_diffusion import FISHGaussianDiffusion
from models.peft import apply_lora_to_model
from models.unet import UNet, SimpleUNet
from utils.dataset import FISHDatasetWithFixedMask
from utils.metrics import batch_evaluate


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


def load_finetuned_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_cfg = checkpoint.get("base_config", checkpoint.get("config", {}))
    ft_cfg = checkpoint.get("finetune_config", {})

    model_type = base_cfg.get("model_type", "unet")
    base_channels = base_cfg.get("base_channels", 64)
    timesteps = int(base_cfg.get("timesteps", 1000))
    loss_type = base_cfg.get("loss_type", "l2")
    mask_strategy = base_cfg.get("mask_strategy", "band")

    model = build_model(model_type=model_type, base_channels=base_channels)

    if "finetune_config" in checkpoint:
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

    diffusion = FISHGaussianDiffusion(
        channels=1,
        timesteps=timesteps,
        loss_type=loss_type,
        mask_strategy=mask_strategy,
    )
    return model, diffusion, base_cfg, ft_cfg


def visualize_comparison(true_matrix, pred_mdm, masked_input, mask, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(true_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[0, 0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(masked_input, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[0, 1].set_title("Masked Input")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(pred_mdm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[1, 0].set_title("Finetuned Prediction")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(mask, cmap="gray", aspect="auto")
    axes[1, 1].set_title("Mask (1=observed, 0=missing)")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test LoRA-finetuned model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Path to test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_vis", type=int, default=10)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bar (less log/console noise when batching many runs).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading finetuned model from {args.checkpoint}...")
    model, diffusion, base_cfg, ft_cfg = load_finetuned_model(args.checkpoint, args.device)
    print("Model loaded successfully")

    if args.data_path is None:
        args.data_path = base_cfg.get("data_path", ft_cfg.get("data_path"))
    if args.data_path is None:
        raise ValueError("data_path is not provided and not found in checkpoint config.")

    seed = int(base_cfg.get("seed", ft_cfg.get("seed", 42)))
    test_ds = FISHDatasetWithFixedMask(args.data_path, split="test", seed=seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test size: {len(test_ds)}")

    sample_masked, _, sample_mask = test_ds[0]
    mask_ratio = (sample_mask == 0).sum().item() / sample_mask.numel()
    print(f"Predefined mask ratio: {mask_ratio:.4f} ({mask_ratio * 100:.2f}%)")

    all_true = []
    all_pred = []
    all_masks = []
    all_masked_input = []

    with torch.no_grad():
        for x_masked, x_target, mask in tqdm(
            test_loader, desc="Finetuned Inference", disable=args.quiet
        ):
            x_masked = x_masked.to(args.device)
            x_target = x_target.to(args.device)
            mask = mask.to(args.device)

            x_imputed = diffusion.impute(model, x_masked, mask)
            all_true.append(x_target.cpu().numpy())
            all_pred.append(x_imputed.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_masked_input.append(x_masked.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_masked_input = np.concatenate(all_masked_input, axis=0)

    metrics = batch_evaluate(all_true, all_pred, all_masks)
    print("\nFinetuned Results:")
    print(f"  PCC (imputed): {metrics.get('pcc_imputed', np.nan):.4f}")
    print(f"  Insulation PCC: {metrics.get('insulation_pcc', np.nan):.4f}")
    print(f"  MSE (imputed): {metrics.get('mse_imputed', np.nan):.6f}")
    print(f"  MAE (imputed): {metrics.get('mae_imputed', np.nan):.6f}")

    np.save(os.path.join(args.output_dir, "true_matrices.npy"), all_true)
    np.save(os.path.join(args.output_dir, "pred_finetuned.npy"), all_pred)
    np.save(os.path.join(args.output_dir, "masks.npy"), all_masks)
    np.save(os.path.join(args.output_dir, "masked_input.npy"), all_masked_input)

    with open(os.path.join(args.output_dir, "metrics_finetuned.json"), "w") as f:
        json.dump(
            {
                "finetuned": {k: float(v) if not isinstance(v, dict) else v for k, v in metrics.items()},
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
        visualize_comparison(
            all_true[i, 0],
            all_pred[i, 0],
            all_masked_input[i, 0],
            all_masks[i, 0],
            os.path.join(vis_dir, f"sample_{i:03d}_finetuned.png"),
        )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
