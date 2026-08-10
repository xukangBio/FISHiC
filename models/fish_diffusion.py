import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FISHGaussianDiffusion(nn.Module):

    def __init__(
        self,
        *,
        channels=1,
        timesteps=1000,
        loss_type='l2',
        mask_strategy='band',
    ):
        super().__init__()
        self.channels = channels
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.mask_strategy = mask_strategy

    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps)

    def create_band_mask(self, batch_size, size, mask_ratio, device):
        masks = []
        n_mask = int(size * mask_ratio)
        
        for _ in range(batch_size):
            mask = torch.ones(size, size, device=device)

            if n_mask > 0:
                missing_bins = torch.randperm(size, device=device)[:n_mask]

                for bin_idx in missing_bins:
                    mask[bin_idx, :] = 0
                    mask[:, bin_idx] = 0

            masks.append(mask)

        return torch.stack(masks).unsqueeze(1)

    def create_patch_mask(self, batch_size, size, mask_ratio, device, patch_size=8):

        assert size % patch_size == 0
        n_patches = size // patch_size
        total_patches = n_patches * n_patches
        n_keep = int(total_patches * (1 - mask_ratio))

        masks = []
        for _ in range(batch_size):
            noise = torch.rand(total_patches, device=device)
            ids_shuffle = torch.argsort(noise)

            patch_mask = torch.zeros(total_patches, device=device)
            patch_mask[ids_shuffle[:n_keep]] = 1
            patch_mask = patch_mask.reshape(n_patches, n_patches)

            full_mask = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
            masks.append(full_mask)

        return torch.stack(masks).unsqueeze(1)

    def q_sample(self, x_start, t):

        B, C, H, W = x_start.shape
        device = x_start.device

        x_t = torch.zeros_like(x_start)
        masks = []

        for i in range(B):
            mask_ratio = int(t[i]) / self.num_timesteps

            if self.mask_strategy == 'band':
                mask = self.create_band_mask(1, H, mask_ratio, device)
            else:
                mask = self.create_patch_mask(1, H, mask_ratio, device)

            masks.append(mask)
            x_t[i] = x_start[i] * mask[0]

        mask_tensor = torch.cat(masks, dim=0)
        return x_t, mask_tensor

    def training_losses(self, model, x_start, t):
        terms = {}

        x_t, mask = self.q_sample(x_start=x_start, t=t)
        output = model(x_t, self._scale_timesteps(t))

        if self.loss_type in ['l1', 'l1+ssim']:
            pixel_loss = (x_start - output).abs()
        elif self.loss_type in ['l2', 'l2+ssim']:
            pixel_loss = (x_start - output) ** 2
        elif self.loss_type == 'ssim':
            pixel_loss = None
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        if pixel_loss is not None:
            mask_inv = 1 - mask  # 1 for masked regions
            pixel_loss = (pixel_loss * mask_inv).sum() / (mask_inv.sum() + 1e-8)
            terms["pixel_loss"] = pixel_loss

        if self.loss_type == 'ssim':
            ssim_term = self.ssim_loss(x_start, output)
            terms["ssim_loss"] = ssim_term
            total_loss = ssim_term
        elif self.loss_type in ['l1+ssim', 'l2+ssim']:
            ssim_term = self.ssim_loss(x_start, output)
            std_term = self.std_loss(output, x_start)
            terms["ssim_loss"] = ssim_term
            terms["std_loss"] = std_term
            lambda_ssim = 1
            lambda_std = 0.1
            total_loss = pixel_loss + lambda_ssim * ssim_term + lambda_std * std_term
        else:
            total_loss = pixel_loss

        terms["loss"] = total_loss
        return terms

    def ssim_loss(self, x, y, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size//2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, window_size, stride=1, padding=window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_xy

        ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim = ssim_n / ssim_d

        return 1 - ssim.mean()

    def std_loss(self, pred, target):

        pred_std = torch.std(pred)
        target_std = torch.std(target)
        return torch.relu(target_std - pred_std)

    @torch.no_grad()
    def impute(self, model, x_masked, mask, num_steps=50):
        """
        Args:
            model: Trained model
            x_masked: Masked input (with missing values set to 0)
            mask: Binary mask (1=observed, 0=missing)
            num_steps: Not used for MDM (kept for compatibility)

        Returns:
            x_imputed: Reconstructed matrix
        """
        B, C, H, W = x_masked.shape
        device = x_masked.device

        mask_2d = mask.view(B, H, W)
        row_sum = mask_2d.sum(dim=-1)
        missing_rows = (row_sum == 0).sum(dim=1)
        missing_ratio = missing_rows.float() / float(H)

        t = (missing_ratio * (self.num_timesteps - 1)).clamp(0, self.num_timesteps - 1)
        t = t.long()

        x_recon = model(x_masked, self._scale_timesteps(t))

        mask_inv = 1 - mask
        x_imputed = x_masked * mask + x_recon * mask_inv

        return x_imputed

    @torch.no_grad()
    def impute_iterative(self, model, x_masked, mask, num_steps=10):
        B, C, H, W = x_masked.shape
        device = x_masked.device

        x_current = x_masked.clone()

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device)

            x_recon = model(x_current, self._scale_timesteps(t))

            mask_inv = 1 - mask
            x_current = x_masked * mask + x_recon * mask_inv

        return x_current
