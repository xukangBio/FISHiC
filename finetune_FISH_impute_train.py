import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet, SimpleUNet
from models.peft import apply_lora_to_model, mark_only_lora_trainable, count_trainable_parameters
from utils.dataset import load_fish_datasets


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


def build_band_mask(batch_size, size, ratio, device):
    masks = []
    n_mask = int(size * ratio)
    for _ in range(batch_size):
        m = torch.ones(size, size, device=device)
        if n_mask > 0:
            missing_bins = torch.randperm(size, device=device)[:n_mask]
            m[missing_bins, :] = 0
            m[:, missing_bins] = 0
        masks.append(m)
    return torch.stack(masks, dim=0).unsqueeze(1)


def compute_t_from_mask(mask, timesteps):
    b, _, h, _ = mask.shape
    mask_2d = mask.view(b, h, h)
    row_sum = mask_2d.sum(dim=-1)
    missing_rows = (row_sum == 0).sum(dim=1).float() / float(h)
    t = (missing_rows * (timesteps - 1)).clamp(0, timesteps - 1).long()
    return t.float() * (1000.0 / timesteps)


def ssim_loss(x, y, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y ** 2, window_size, stride=1, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_xy
    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return 1 - ssim.mean()


def std_loss(pred, target):
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    return torch.relu(target_std - pred_std)


def train_one_epoch(
    model, loader, optimizer, device, timesteps,
    min_ratio, max_ratio, obs_weight,
    lambda_ssim, lambda_std,
):
    model.train()
    mse = nn.MSELoss(reduction="mean")
    losses = []
    miss_losses = []
    obs_losses = []
    ssim_losses = []
    std_losses = []

    pbar = tqdm(loader, desc="Fine-tune")
    for _, x_target, _ in pbar:
        x_target = x_target.to(device, non_blocking=True)
        b, _, h, _ = x_target.shape

        ratio = float(np.random.uniform(min_ratio, max_ratio))
        mask = build_band_mask(b, h, ratio, device)
        x_masked = x_target * mask

        t_scaled = compute_t_from_mask(mask, timesteps)
        pred = model(x_masked, t_scaled)

        miss = 1.0 - mask
        loss_missing = ((pred - x_target) ** 2 * miss).sum() / (miss.sum() + 1e-8)
        loss_observed = mse(pred * mask, x_target * mask)
        ssim_term = ssim_loss(x_target, pred)
        std_term = std_loss(pred, x_target)
        loss = (
            loss_missing
            + obs_weight * loss_observed
            + lambda_ssim * ssim_term
            + lambda_std * std_term
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        miss_losses.append(loss_missing.item())
        obs_losses.append(loss_observed.item())
        ssim_losses.append(ssim_term.item())
        std_losses.append(std_term.item())
        pbar.set_postfix(
            loss=f"{np.mean(losses):.6f}",
            miss=f"{np.mean(miss_losses):.6f}",
            ssim=f"{np.mean(ssim_losses):.4f}",
            std=f"{np.mean(std_losses):.4f}",
        )

    return {
        "loss": float(np.mean(losses)),
        "missing_loss": float(np.mean(miss_losses)),
        "observed_loss": float(np.mean(obs_losses)),
        "ssim_loss": float(np.mean(ssim_losses)),
        "std_loss": float(np.mean(std_losses)),
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning (supervised masked->full)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--multi_gpu", action="store_true", help="Use all visible GPUs via DataParallel")

    # Mask setting for supervised finetune
    parser.add_argument("--min_mask_ratio", type=float, default=0.4)
    parser.add_argument("--max_mask_ratio", type=float, default=0.6)
    parser.add_argument("--observed_loss_weight", type=float, default=0.1)
    parser.add_argument("--lambda_ssim", type=float, default=1.0, help="SSIM loss weight")
    parser.add_argument("--lambda_std", type=float, default=0.1, help="Std loss weight")

    # LoRA setting
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="",
        help="comma-separated module name keywords, empty means all Conv2d/Linear",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.output_dir, "finetune_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "unet")
    base_channels = cfg.get("base_channels", 64)
    timesteps = int(cfg.get("timesteps", 1000))

    print(f"Load pretrained: {args.pretrained_checkpoint}")
    print(f"Base config: model_type={model_type}, base_channels={base_channels}, timesteps={timesteps}")

    model = build_model(model_type=model_type, base_channels=base_channels)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("Loaded pretrained weights.")

    target_modules = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    replaced = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules if len(target_modules) > 0 else None,
    )
    mark_only_lora_trainable(model)
    trainable, total = count_trainable_parameters(model)
    print(f"LoRA injected modules: {replaced}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")

    model = model.to(args.device)
    n_gpu = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    if n_gpu > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpu} GPUs")

    (train_ds,) = load_fish_datasets(args.data_path, seed=args.seed, splits=("train",))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Train size: {len(train_ds)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    history = []
    for epoch in range(1, args.num_epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            timesteps=timesteps,
            min_ratio=args.min_mask_ratio,
            max_ratio=args.max_mask_ratio,
            obs_weight=args.observed_loss_weight,
            lambda_ssim=args.lambda_ssim,
            lambda_std=args.lambda_std,
        )
        history.append({"epoch": epoch, **metrics})
        print(
            f"Epoch {epoch}/{args.num_epochs} "
            f"loss={metrics['loss']:.6f} miss={metrics['missing_loss']:.6f} "
            f"obs={metrics['observed_loss']:.6f} ssim={metrics['ssim_loss']:.4f} std={metrics['std_loss']:.4f}"
        )

        save_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": save_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "finetune_loss": metrics["loss"],
                    "base_checkpoint": args.pretrained_checkpoint,
                    "base_config": cfg,
                    "finetune_config": vars(args),
                },
                os.path.join(args.output_dir, "best_finetuned.pt"),
            )
            print(f"Saved best finetuned model with loss={best_loss:.6f}")

        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": save_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "base_checkpoint": args.pretrained_checkpoint,
                    "base_config": cfg,
                    "finetune_config": vars(args),
                },
                os.path.join(args.output_dir, f"finetune_epoch_{epoch}.pt"),
            )

    with open(os.path.join(args.output_dir, "finetune_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Fine-tuning finished. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
