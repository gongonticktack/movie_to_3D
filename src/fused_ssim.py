from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(channel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.arange(11, dtype=dtype, device=device) - 5
    kernel_1d = torch.exp(-(coords**2) / 8.0)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.expand(channel, 1, 11, 11).contiguous()


def fused_ssim(img1: torch.Tensor, img2: torch.Tensor, padding: str = "same", train: bool = True) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")
    if img1.ndim != 4:
        raise ValueError("img1 and img2 must be NCHW tensors")
    if padding not in {"same", "valid"}:
        raise ValueError("padding must be 'same' or 'valid'")

    channel = img1.shape[1]
    window = _gaussian_window(channel, img1.dtype, img1.device)
    pad = 5 if padding == "same" else 0

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()
