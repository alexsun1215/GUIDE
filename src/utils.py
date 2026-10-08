import os
import math
import random
import logging

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("guide")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(out_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def fftn_safe(x: torch.Tensor, dim):
    if x.device.type == "mps":
        return torch.fft.fftn(x.float().cpu(), dim=dim).to(x.device)
    return torch.fft.fftn(x, dim=dim)


def ifftn_safe(x: torch.Tensor, dim):
    if x.device.type == "mps":
        return torch.fft.ifftn(x.cpu(), dim=dim).to(x.device)
    return torch.fft.ifftn(x, dim=dim)


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def sinkhorn_plan(cost: torch.Tensor, eps: float = 0.05, iters: int = 16) -> torch.Tensor:
    K = torch.exp(-cost / max(eps, 1e-6))
    n, m = cost.shape[-2], cost.shape[-1]
    a = cost.new_full(cost.shape[:-1], 1.0 / n)
    b = cost.new_full(cost.shape[:-2] + (m,), 1.0 / m)
    u = torch.ones_like(a)
    for _ in range(iters):
        kv = torch.einsum("...nm,...m->...n", K, b / (torch.einsum("...nm,...n->...m", K, u) + 1e-9))
        u = a / (kv + 1e-9)
    v = b / (torch.einsum("...nm,...n->...m", K, u) + 1e-9)
    return u.unsqueeze(-1) * K * v.unsqueeze(-2)


def sinkhorn_cost(cost: torch.Tensor, eps: float = 0.05, iters: int = 16) -> torch.Tensor:
    plan = sinkhorn_plan(cost, eps, iters)
    return (plan * cost).flatten(-2).sum(-1)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    sqc = math.sqrt(max(c, 1e-6))
    vn = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.tanh(sqc * vn) * v / (sqc * vn)


def logmap0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    sqc = math.sqrt(max(c, 1e-6))
    xn = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.atanh((sqc * xn).clamp(-1 + 1e-5, 1 - 1e-5)) * x / (sqc * xn)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    num = 2.0 * c * (x - y).pow(2).sum(-1)
    den = (1.0 - c * x.pow(2).sum(-1)).clamp_min(1e-9) * (1.0 - c * y.pow(2).sum(-1)).clamp_min(1e-9)
    return torch.acosh((1.0 + num / den).clamp_min(1.0 + 1e-7)) / math.sqrt(max(c, 1e-6))


def knn_adjacency(feats: torch.Tensor, k: int = 8) -> torch.Tensor:
    d = torch.cdist(feats, feats)
    k = min(k, feats.shape[-2] - 1)
    idx = d.topk(k + 1, dim=-1, largest=False).indices[..., 1:]
    adj = torch.zeros_like(d)
    adj.scatter_(-1, idx, 1.0)
    return ((adj + adj.transpose(-1, -2)) > 0).to(feats.dtype)


def normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(-1).clamp_min(1e-9)
    dinv = deg.pow(-0.5)
    eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype).expand_as(adj)
    return eye - dinv.unsqueeze(-1) * adj * dinv.unsqueeze(-2)


def chebyshev_basis(lap: torch.Tensor, x: torch.Tensor, order: int):
    out = [x, torch.einsum("...nm,...mc->...nc", lap, x)]
    for _ in range(2, order):
        out.append(2.0 * torch.einsum("...nm,...mc->...nc", lap, out[-1]) - out[-2])
    return out


def radial_spectrum(mag: torch.Tensor, n_bins: int) -> torch.Tensor:
    d, h, w = mag.shape[-3:]
    fz = torch.fft.fftfreq(d, device=mag.device).abs()
    fy = torch.fft.fftfreq(h, device=mag.device).abs()
    fx = torch.fft.fftfreq(w, device=mag.device).abs()
    rho = torch.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    rho = (rho / rho.max().clamp_min(1e-9) * (n_bins - 1)).round().long().clamp(0, n_bins - 1)
    flat = rho.flatten()
    m = mag.flatten(-3)
    out = m.new_zeros(m.shape[:-1] + (n_bins,))
    out.index_add_(-1, flat, m)
    cnt = torch.bincount(flat, minlength=n_bins).clamp_min(1).to(out.dtype)
    return out / cnt
