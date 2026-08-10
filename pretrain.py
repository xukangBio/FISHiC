import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')

from models.unet import UNet, SimpleUNet
from models.fish_diffusion import FISHGaussianDiffusion
from utils.dataset import load_fish_datasets
from utils.metrics import batch_evaluate


def _json_default(o):
    if isinstance(o, (np.floating, np.float32, np.float64)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


def _to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (float, int, str, bool, type(None))):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    return obj



def train_epoch(model, diffusion, dataloader, optimizer, device, epoch):
    model.train()
    n_batches = 0
    accum = {}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        x_masked, x_target, mask = batch
        x_target = x_target.to(device)

        B = x_target.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()

        optimizer.zero_grad()
        losses = diffusion.training_losses(model, x_target, t)
        loss = losses['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_batches += 1
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            accum[k] = accum.get(k, 0.0) + v

        pbar.set_postfix({'loss': loss.item()})

    return {k: v / n_batches for k, v in accum.items()}


def validate(model, diffusion, dataloader, device, max_samples=100):
    model.eval()
    total_loss = 0
    n_batches = 0

    all_true = []
    all_pred = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            x_masked, x_target, mask = batch
            x_masked = x_masked.to(device)
            x_target = x_target.to(device)
            mask = mask.to(device)

            B = x_target.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()

            losses = diffusion.training_losses(model, x_target, t)
            loss = losses['loss']
            total_loss += loss.item()
            n_batches += 1

            if len(all_true) < max_samples:
                x_imputed = diffusion.impute(model, x_masked, mask)

                all_true.append(x_target.cpu().numpy())
                all_pred.append(x_imputed.cpu().numpy())
                all_masks.append(mask.cpu().numpy())

    if len(all_true) > 0:
        all_true = np.concatenate(all_true, axis=0)[:max_samples]
        all_pred = np.concatenate(all_pred, axis=0)[:max_samples]
        all_masks = np.concatenate(all_masks, axis=0)[:max_samples]

        metrics = batch_evaluate(all_true, all_pred, all_masks)
    else:
        metrics = {}

    metrics['val_loss'] = total_loss / n_batches
    return metrics


def test(model, diffusion, dataloader, device, output_dir=None):
    model.eval()

    all_true = []
    all_pred = []
    all_masks = []
    all_masked_input = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            x_masked, x_target, mask = batch
            x_masked = x_masked.to(device)
            x_target = x_target.to(device)
            mask = mask.to(device)

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

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'true_matrices.npy'), all_true)
        np.save(os.path.join(output_dir, 'pred_matrices.npy'), all_pred)
        np.save(os.path.join(output_dir, 'masks.npy'), all_masks)
        np.save(os.path.join(output_dir, 'masked_input.npy'), all_masked_input)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) if not isinstance(v, dict) else v
                      for k, v in metrics.items()}, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train MDM for DNA FISH imputation (Predefined Masks)')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to folder containing train/valid/test (valid 与 test 需含 distance_mask.npy)')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='unet',
                       choices=['unet', 'simple_unet'],
                       help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--loss_type', type=str, default='l2',
                       choices=['l1', 'l2', 'ssim', 'l1+ssim', 'l2+ssim'],
                       help='Loss function type')
    parser.add_argument('--mask_strategy', type=str, default='band',
                       choices=['band', 'patch'],
                       help='Masking strategy')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Model save interval (epochs), save checkpoint every N epochs')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Arguments: {args}")
    print(f"Using device: {args.device}")

    print("Loading data (train only, full graph)...")
    (train_ds,) = load_fish_datasets(args.data_path, seed=args.seed, splits=('train',))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Train size: {len(train_ds)}")

    sample_masked, sample_full, sample_mask = train_ds[0]
    mask_ratio = (sample_mask == 0).sum().item() / sample_mask.numel()
    print(f"Train mask ratio (masked): {mask_ratio:.4f} ({mask_ratio*100:.2f}%) (0 = full graph)")

    matrix_size = sample_masked.shape[-1]
    print(f"Matrix size: {matrix_size}x{matrix_size}")

    print("Creating model...")
    if args.model_type == 'unet':
        model = UNet(
            in_channels=1,
            base_channels=args.base_channels,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=256,
            dropout=0.1,
            use_attention=True,
        )
    else:
        model = SimpleUNet(
            in_channels=1,
            base_channels=args.base_channels,
            time_emb_dim=128,
        )

    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    diffusion = FISHGaussianDiffusion(
        channels=1,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
        mask_strategy=args.mask_strategy,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                          patience=10, verbose=True)

    print("Starting training...")
    best_train_loss = float('inf')
    results = []

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(model, diffusion, train_loader, optimizer, args.device, epoch)
        train_loss = train_metrics['loss']
        epoch_record = {'epoch': epoch, 'train_loss': train_loss}
        for k in ('pixel_loss', 'ssim_loss', 'std_loss'):
            if k in train_metrics:
                epoch_record[f'train_{k}'] = train_metrics[k]
        results.append(epoch_record)

        print(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.6f}")

        scheduler.step(train_loss)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': vars(args),
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with train_loss={train_loss:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(
            _to_json_serializable(results),
            f,
            indent=2,
            default=_json_default,
        )

    print(f"\nTraining finished. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
