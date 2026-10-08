from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PCADConfig
from ..utils import sinkhorn_plan, radial_spectrum


class SinkhornPhaseBridge(nn.Module):
    def __init__(self, channels: int, tokens: int = 6, eps: float = 0.05, iters: int = 16):
        super().__init__()
        self.tokens = tokens
        self.eps = eps
        self.iters = iters
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _tok(self, x):
        x = F.adaptive_avg_pool3d(x, self.tokens)
        return x.flatten(2).transpose(1, 2)

    def forward(self, feat_nc: torch.Tensor, feat_pv: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c = feat_nc.shape[:2]
        qn = F.normalize(self._tok(self.q(feat_nc)), dim=-1)
        kp = F.normalize(self._tok(self.k(feat_pv)), dim=-1)
        cost = 1.0 - torch.bmm(qn, kp.transpose(1, 2))
        plan = sinkhorn_plan(cost, self.eps, self.iters)
        pv_tok = self._tok(feat_pv)
        transported = torch.bmm(plan * plan.shape[1], pv_tok)
        t = self.tokens
        bridged = transported.transpose(1, 2).reshape(b, c, t, t, t)
        bridged = F.interpolate(bridged, size=feat_nc.shape[2:], mode="trilinear", align_corners=False)
        loss_bridge = (plan * cost).flatten(1).sum(-1).mean()
        return {"transported": feat_nc + self.gamma * bridged, "loss_bridge": loss_bridge}


class CrossPhaseMutualInformation(nn.Module):
    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.t = nn.Sequential(
            nn.Linear(2 * channels, hidden), nn.ELU(inplace=True),
            nn.Linear(hidden, hidden), nn.ELU(inplace=True), nn.Linear(hidden, 1),
        )

    def forward(self, feat_nc: torch.Tensor, feat_pv: torch.Tensor) -> torch.Tensor:
        a = F.adaptive_avg_pool3d(feat_nc, 1).flatten(1)
        p = F.adaptive_avg_pool3d(feat_pv, 1).flatten(1)
        joint = self.t(torch.cat([a, p], dim=1))
        shuffle = p[torch.randperm(p.shape[0], device=p.device)]
        marginal = self.t(torch.cat([a, shuffle], dim=1))
        return -(joint.mean() - (torch.logsumexp(marginal, dim=0) - torch.log(torch.tensor(float(p.shape[0])))))


class SpectralPhaseConsistency(nn.Module):
    def __init__(self, bands: int = 6):
        super().__init__()
        self.bands = bands
        self.scale = nn.Parameter(torch.ones(bands))

    def forward(self, gain: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gm = torch.fft.fftn(gain.float(), dim=(2, 3, 4)).abs().mean(1)
        rm = torch.fft.fftn(residual.float(), dim=(2, 3, 4)).abs().mean(1)
        sg = radial_spectrum(gm, self.bands)
        sr = radial_spectrum(rm, self.bands)
        sg = sg / sg.sum(-1, keepdim=True).clamp_min(1e-9)
        sr = sr / sr.sum(-1, keepdim=True).clamp_min(1e-9)
        return ((self.scale * (sg - sr)) ** 2).sum(-1).mean()


def _gaussian_kernel3d(sigma: float = 1.0, ksize: int = 5) -> torch.Tensor:
    ax = torch.arange(ksize) - (ksize - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    k = k / k.sum()
    return k.view(1, 1, ksize, ksize, ksize)


def _build_base_grid(b, d, h, w, device, dtype):
    zs = torch.linspace(-1, 1, d, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid = torch.stack([gx, gy, gz], dim=-1)
    return grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)


class PCAD(nn.Module):
    """Phase-Contrast Anatomically-anchored Distillation."""

    def __init__(self, cfg: PCADConfig, feat_channels: int, max_disp: float = 0.3):
        super().__init__()
        self.cfg = cfg
        self.max_disp = max_disp
        c = feat_channels
        h = cfg.deform_hidden
        self.flow_net = nn.Sequential(
            nn.Conv3d(2 * c, h, 3, padding=1), nn.GroupNorm(8, h), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h, h, 3, padding=1), nn.GroupNorm(8, h), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h, 3, 3, padding=1),
        )
        self.flow_net[-1].weight.data.normal_(0, 1e-4)
        self.flow_net[-1].bias.data.zero_()
        self.gain_head = nn.Sequential(
            nn.Conv3d(c, h, 3, padding=1), nn.GroupNorm(8, h), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h, c, 3, padding=1),
        )
        self.gain_attn = nn.Sequential(nn.Conv3d(c, 1, 1), nn.Sigmoid())
        self.register_buffer("gk", _gaussian_kernel3d(1.0, 5))
        if cfg.use_sinkhorn_bridge:
            self.phase_bridge = SinkhornPhaseBridge(c, cfg.bridge_tokens, cfg.sinkhorn_eps, cfg.sinkhorn_iters)
        if cfg.use_cross_phase_mi:
            self.phase_mi = CrossPhaseMutualInformation(c, cfg.mi_hidden)
        if cfg.use_spectral_consistency:
            self.spectral_consistency = SpectralPhaseConsistency(cfg.spectral_bands)

    def warp(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = feat.shape
        base = _build_base_grid(b, d, h, w, feat.device, feat.dtype)
        disp = torch.stack([flow[:, 2], flow[:, 1], flow[:, 0]], dim=-1)
        grid = base + disp
        return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @staticmethod
    def _smooth_loss(flow: torch.Tensor) -> torch.Tensor:
        dz = flow[:, :, 1:] - flow[:, :, :-1]
        dy = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
        return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3.0

    @staticmethod
    def _jacobian_dev(flow_vox: torch.Tensor) -> torch.Tensor:
        d = flow_vox
        dD = d[:, :, 1:, :-1, :-1] - d[:, :, :-1, :-1, :-1]
        dH = d[:, :, :-1, 1:, :-1] - d[:, :, :-1, :-1, :-1]
        dW = d[:, :, :-1, :-1, 1:] - d[:, :, :-1, :-1, :-1]
        colD = dD.clone(); colD[:, 0] += 1.0
        colH = dH.clone(); colH[:, 1] += 1.0
        colW = dW.clone(); colW[:, 2] += 1.0
        det = (
            colD[:, 0] * (colH[:, 1] * colW[:, 2] - colH[:, 2] * colW[:, 1])
            - colD[:, 1] * (colH[:, 0] * colW[:, 2] - colH[:, 2] * colW[:, 0])
            + colD[:, 2] * (colH[:, 0] * colW[:, 1] - colH[:, 1] * colW[:, 0])
        )
        dev = (det - 1.0).abs().unsqueeze(1)
        return F.pad(dev, (0, 1, 0, 1, 0, 1), mode="replicate")

    def _anchor_prox(self, anchor: torch.Tensor, size) -> torch.Tensor:
        a = F.interpolate(anchor, size=size, mode="trilinear", align_corners=False)
        a = F.conv3d(a, self.gk, padding=self.gk.shape[-1] // 2)
        a = a / (a.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
        return a.clamp(0, 1)

    def forward(self, feat_nc: torch.Tensor, feat_pv: torch.Tensor,
                anchor: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        size = feat_nc.shape[2:]
        flow = torch.tanh(self.flow_net(torch.cat([feat_nc, feat_pv], dim=1))) * self.max_disp
        warped_pv = self.warp(feat_pv, flow)

        residual = (warped_pv - feat_nc).detach()
        gain = self.gain_head(feat_nc)
        gain_attn = self.gain_attn(feat_nc)

        res_energy = (warped_pv - feat_nc).pow(2).mean(dim=1, keepdim=True)
        smooth_map = (flow[:, :, 1:] - flow[:, :, :-1]).pow(2).mean(dim=1, keepdim=True)
        smooth_map = F.pad(smooth_map, (0, 0, 0, 0, 0, 1), mode="replicate")
        flow_vox = flow * torch.tensor([s / 2.0 for s in size], device=flow.device).view(1, 3, 1, 1, 1)
        jac_dev = self._jacobian_dev(flow_vox)
        conf = torch.sigmoid(-(self.cfg.gate_alpha_smooth * smooth_map
                               + self.cfg.gate_alpha_residual * res_energy.detach()
                               + self.cfg.gate_alpha_jac * jac_dev))

        if anchor is not None:
            prox = self._anchor_prox(anchor, size)
        else:
            prox = torch.ones_like(conf)
        gate = conf * torch.exp(-(1.0 - prox) / self.cfg.anchor_tau)

        alpha = 1.0 + self.cfg.anchor_weight * prox
        align_res = (warped_pv - feat_nc).pow(2).mean(dim=1, keepdim=True)
        sample_w = valid.view(-1, 1, 1, 1, 1) if valid is not None else torch.ones(feat_nc.size(0), 1, 1, 1, 1, device=feat_nc.device)
        denom = sample_w.sum() * align_res[0, 0].numel() + 1e-6
        loss_align = (sample_w * (alpha * align_res)).sum() / denom
        loss_align = loss_align + self.cfg.lambda_smooth * self._smooth_loss(flow)
        loss_align = loss_align + self.cfg.lambda_anchor * (sample_w * (prox * align_res)).sum() / denom

        distill_err = (gain - residual).pow(2).mean(dim=1, keepdim=True)
        w = (gate * sample_w).detach()
        loss_distill = (w * distill_err).sum() / (w.sum() + 1e-6)

        out = {
            "flow": flow,
            "warped_pv": warped_pv,
            "gain": gain,
            "gain_attn": gain_attn,
            "gate": gate,
            "loss_align": loss_align,
            "loss_distill": loss_distill,
        }
        if self.cfg.use_sinkhorn_bridge:
            br = self.phase_bridge(feat_nc, warped_pv)
            out["bridge_feat"] = br["transported"]
            out["loss_bridge"] = self.cfg.lambda_bridge * br["loss_bridge"]
        if self.cfg.use_cross_phase_mi:
            out["loss_mi"] = self.cfg.lambda_mi * self.phase_mi(feat_nc, feat_pv)
        if self.cfg.use_spectral_consistency:
            out["loss_spectral"] = self.cfg.lambda_spectral * self.spectral_consistency(gain, residual)
        return out
