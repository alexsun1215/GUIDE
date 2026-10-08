from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AFDGConfig
from ..utils import fftn_safe, ifftn_safe, radial_spectrum


class WaveletScattering(nn.Module):
    def __init__(self, channels: int, scales: int = 3, order: int = 2):
        super().__init__()
        self.scales = scales
        self.order = order
        self.proj = nn.Conv3d(channels * scales, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _lowpass(self, x: torch.Tensor, frac: float) -> torch.Tensor:
        d, h, w = x.shape[-3:]
        fz = torch.fft.fftfreq(d, device=x.device).abs()
        fy = torch.fft.fftfreq(h, device=x.device).abs()
        fx = torch.fft.fftfreq(w, device=x.device).abs()
        rho = torch.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
        mask = (rho <= frac).float().view(1, 1, d, h, w)
        return torch.fft.ifftn(torch.fft.fftn(x.float(), dim=(2, 3, 4)) * mask, dim=(2, 3, 4)).real

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        coeffs = []
        cur = feat
        mod = feat
        for s in range(self.scales):
            lp = self._lowpass(cur, 0.5 / (2 ** s))
            mod = (cur - lp).abs()
            coeffs.append(lp)
            cur = mod
        sc = self.proj(torch.cat(coeffs, dim=1))
        return {"feat": feat + self.gamma * sc, "loss_scatter": mod.mean()}


class HarmonicDescriptor(nn.Module):
    def __init__(self, n_harmonics: int = 9):
        super().__init__()
        self.n = n_harmonics
        self.mlp = nn.Sequential(
            nn.Linear(n_harmonics, n_harmonics), nn.GELU(), nn.Linear(n_harmonics, n_harmonics),
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        mag = torch.fft.fftn(feat.float(), dim=(2, 3, 4)).abs().mean(1)
        spec = radial_spectrum(mag, self.n)
        desc = self.mlp(spec)
        flat = torch.exp(spec.clamp_min(1e-9).log().mean(-1)) / spec.mean(-1).clamp_min(1e-9)
        return {"descriptor": desc, "loss_harmonic": (1.0 - flat).mean()}


class AFDG(nn.Module):
    """Anatomy-anchored Frequency-Domain Generalization (lite)."""

    def __init__(self, cfg: AFDGConfig, feat_channels: int, num_classes: int = 3):
        super().__init__()
        self.cfg = cfg
        hidden = max(num_classes, feat_channels // 2)
        groups = 8 if hidden % 8 == 0 else 1
        self.head = nn.Sequential(
            nn.Conv3d(feat_channels, hidden, 3, stride=2, padding=1), nn.GroupNorm(groups, hidden), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, stride=2, padding=1), nn.GroupNorm(groups, hidden), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(hidden, num_classes),
        )
        self._mask_cache: Dict = {}
        if cfg.use_wavelet_scattering:
            self.scattering = WaveletScattering(feat_channels, cfg.scattering_scales, cfg.scattering_order)
        if cfg.use_harmonic_descriptors:
            self.harmonic = HarmonicDescriptor(cfg.n_harmonics)

    def _lowpass(self, size, device) -> torch.Tensor:
        key = (size, device.type)
        if key in self._mask_cache:
            return self._mask_cache[key]
        d, h, w = size
        fz = torch.fft.fftfreq(d, device=device).abs() / 0.5
        fy = torch.fft.fftfreq(h, device=device).abs() / 0.5
        fx = torch.fft.fftfreq(w, device=device).abs() / 0.5
        rho = torch.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
        mask = (rho <= self.cfg.cutoff_ratio).float()
        mask = mask.view(1, 1, d, h, w)
        self._mask_cache[key] = mask
        return mask

    def decompose(self, feat: torch.Tensor):
        size = feat.shape[2:]
        lp = self._lowpass(size, feat.device)
        spec = fftn_safe(feat, dim=(2, 3, 4))
        low = ifftn_safe(spec * lp, dim=(2, 3, 4)).real
        high = feat - low
        return low, high

    @staticmethod
    def _cross_center_perm(center_id: torch.Tensor) -> torch.Tensor:
        b = center_id.shape[0]
        perm = torch.randperm(b, device=center_id.device)
        for i in range(b):
            if center_id[perm[i]] == center_id[i]:
                for j in range(b):
                    if center_id[perm[j]] != center_id[i] and center_id[perm[i]] != center_id[j]:
                        perm[i], perm[j] = perm[j].clone(), perm[i].clone()
                        break
        return perm

    def forward(self, feat: torch.Tensor, center_id: torch.Tensor,
                organ: Optional[torch.Tensor] = None,
                organ_valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        low, high = self.decompose(feat)
        logits_orig = self.head(low + high)
        perm = self._cross_center_perm(center_id)
        swapped = low + high[perm]
        logits_swap = self.head(swapped)

        p_orig = F.log_softmax(logits_orig, dim=1)
        p_swap = F.log_softmax(logits_swap, dim=1)
        loss_freq_cons = 0.5 * (
            F.kl_div(p_swap, p_orig.exp(), reduction="batchmean")
            + F.kl_div(p_orig, p_swap.exp(), reduction="batchmean")
        )

        if organ is not None:
            m = F.interpolate(organ, size=feat.shape[2:], mode="trilinear", align_corners=False)
            energy = high.pow(2).mean(dim=1, keepdim=True)
            outside = energy * (1.0 - m)
            if organ_valid is not None:
                v = organ_valid.view(-1, 1, 1, 1, 1)
                loss_spatial = (outside * v).sum() / (v.sum() * outside[0, 0].numel() + 1e-6)
            else:
                loss_spatial = outside.mean()
        else:
            loss_spatial = feat.new_zeros(())

        out = {
            "low": low,
            "high": high,
            "logits_afdg": logits_orig,
            "loss_freq_cons": loss_freq_cons,
            "loss_spatial": loss_spatial,
        }
        if self.cfg.use_wavelet_scattering:
            sc = self.scattering(high)
            out["scatter_feat"] = sc["feat"]
            out["loss_scatter"] = self.cfg.lambda_scatter * sc["loss_scatter"]
        if self.cfg.use_harmonic_descriptors:
            hd = self.harmonic(high)
            out["harmonic_descriptor"] = hd["descriptor"]
            out["loss_harmonic"] = self.cfg.lambda_harmonic * hd["loss_harmonic"]
        return out
