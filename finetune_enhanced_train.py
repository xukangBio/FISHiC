import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.peft import apply_lora_to_model, count_trainable_parameters, mark_only_lora_trainable
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


def ssim_loss(x, y, window_size=11):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y ** 2, window_size, stride=1, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_xy
    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return 1.0 - ssim.mean()


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


def train_one_epoch(model, loader, optimizer, device, lambda_l1, lambda_l2, lambda_ssim):
    model.train()
    running = {"loss": [], "l1": [], "l2": [], "ssim": []}
    pbar = tqdm(loader, desc="Enhance finetune")
    for x_in, y in pbar:
        x_in = x_in.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = forward_model(model, x_in)
        l1_term = F.l1_loss(pred, y)
        l2_term = F.mse_loss(pred, y)
        ssim_term = ssim_loss(pred, y)
        loss = lambda_l1 * l1_term + lambda_l2 * l2_term + lambda_ssim * ssim_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["loss"].append(loss.item())
        running["l1"].append(l1_term.item())
        running["l2"].append(l2_term.item())
        running["ssim"].append(ssim_term.item())
        pbar.set_postfix(loss=f"{np.mean(running['loss']):.6f}", l1=f"{np.mean(running['l1']):.6f}")

    return {k: float(np.mean(v)) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    for x_in, y in loader:
        x_in = x_in.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = forward_model(model, x_in)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    return evaluate_regression_metrics(all_true, all_pred)


def main():
    parser = argparse.ArgumentParser(description="Downstream finetune: normalized scHi-C -> normalized enhanced scHi-C")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_filename", type=str, default="normalized_distance_matrices.npy")
    parser.add_argument("--label_filename", type=str, default="normalized_enhanced_distance_matrices.npy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_ssim", type=float, default=0.5)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA and finetune full model params")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.output_dir, "finetune_enhanced_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "unet")
    base_channels = int(cfg.get("base_channels", 64))
    print(f"Base model config: model_type={model_type}, base_channels={base_channels}")

    model = build_model(model_type=model_type, base_channels=base_channels)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("Loaded pretrained weights.")

    if args.no_lora:
        trainable = sum(p.numel() for p in model.parameters())
        total = trainable
        print("LoRA disabled: full-parameter finetuning.")
    else:
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

    train_ds = PairedNormalizedHiCDataset(
        args.data_path,
        split="train",
        input_filename=args.input_filename,
        label_filename=args.label_filename,
    )
    val_ds = PairedNormalizedHiCDataset(
        args.data_path,
        split="val",
        input_filename=args.input_filename,
        label_filename=args.label_filename,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_mse = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            lambda_l1=args.lambda_l1,
            lambda_l2=args.lambda_l2,
            lambda_ssim=args.lambda_ssim,
        )
        val_metrics = validate(model, val_loader, args.device)
        record = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(
            f"Epoch {epoch}/{args.num_epochs} "
            f"loss={train_metrics['loss']:.6f} "
            f"val_mse={val_metrics['mse_all']:.6f} "
            f"val_mae={val_metrics['mae_all']:.6f} "
            f"val_pcc={val_metrics['pcc_all']:.4f}"
        )

        save_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        if val_metrics["mse_all"] < best_mse:
            best_mse = val_metrics["mse_all"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": save_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mse": best_mse,
                    "base_checkpoint": args.pretrained_checkpoint,
                    "base_config": cfg,
                    "finetune_config": vars(args),
                    "task": "normalized_hic_to_normalized_enhanced_hic",
                },
                os.path.join(args.output_dir, "best_finetuned_enhanced.pt"),
            )
            print(f"Saved best model with val_mse={best_mse:.6f}")

        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": save_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "base_checkpoint": args.pretrained_checkpoint,
                    "base_config": cfg,
                    "finetune_config": vars(args),
                    "task": "normalized_hic_to_normalized_enhanced_hic",
                },
                os.path.join(args.output_dir, f"finetune_enhanced_epoch_{epoch}.pt"),
            )

    with open(os.path.join(args.output_dir, "finetune_enhanced_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Finished. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
