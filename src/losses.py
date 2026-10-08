from typing import Dict, Set
import math

import torch
import torch.nn.functional as F

from .config import Config


def kl_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    beta = torch.ones_like(alpha)
    s_a = alpha.sum(1, keepdim=True)
    s_b = beta.sum(1, keepdim=True)
    lnb = torch.lgamma(s_a) - torch.lgamma(alpha).sum(1, keepdim=True)
    lnb_uni = torch.lgamma(beta).sum(1, keepdim=True) - torch.lgamma(s_b)
    dg0 = torch.digamma(s_a)
    dg1 = torch.digamma(alpha)
    kl = ((alpha - beta) * (dg1 - dg0)).sum(1, keepdim=True) + lnb + lnb_uni
    return kl.squeeze(1)


def edl_loss(alpha: torch.Tensor, label: torch.Tensor, anneal: float) -> torch.Tensor:
    y = F.one_hot(label.clamp(min=0), num_classes=alpha.shape[1]).float()
    s = alpha.sum(1, keepdim=True)
    p = alpha / s
    err = ((y - p) ** 2).sum(1)
    var = (p * (1 - p) / (s + 1)).sum(1)
    alpha_tilde = y + (1 - y) * alpha
    kl = kl_dirichlet(alpha_tilde)
    return (err + var + anneal * kl).mean()


def _soft_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3, 4)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return 1.0 - (2 * inter + 1.0) / (denom + 1.0)


def seg_loss(seg_logits: torch.Tensor, target: torch.Tensor, has_lesion: torch.Tensor) -> torch.Tensor:
    w = has_lesion.view(-1)
    if w.sum() < 1:
        return seg_logits.new_zeros(())
    prob = torch.sigmoid(seg_logits)
    bce = F.binary_cross_entropy_with_logits(seg_logits, target, reduction="none").mean((1, 2, 3, 4))
    dice = _soft_dice(prob, target)
    per = bce + dice
    return (per * w).sum() / (w.sum() + 1e-6)


def size_loss(size_pred: torch.Tensor, dmax_cm: torch.Tensor, has_lesion: torch.Tensor) -> torch.Tensor:
    valid = (has_lesion.view(-1) > 0.5) & (dmax_cm.view(-1) > 0)
    if valid.sum() < 1:
        return size_pred.new_zeros(())
    return F.smooth_l1_loss(size_pred[valid], dmax_cm.view(-1)[valid])


def mask_equiv_diameter_cm(seg_logits: torch.Tensor, voxel_size_cm: float) -> torch.Tensor:
    prob = torch.sigmoid(seg_logits)
    v_vox = prob.sum((1, 2, 3, 4))
    v_cm3 = v_vox * (voxel_size_cm ** 3)
    return 2.0 * (3.0 * v_cm3 / (4.0 * math.pi) + 1e-6) ** (1.0 / 3.0)


def consistency_loss(seg_logits, size_pred, batch, cfg: Config) -> torch.Tensor:
    valid = (batch["has_lesion"].view(-1) > 0.5)
    if valid.sum() < 1:
        return size_pred.new_zeros(())
    d_mask = mask_equiv_diameter_cm(seg_logits, cfg.data.voxel_size_cm)
    return F.smooth_l1_loss(size_pred[valid], d_mask[valid])


def cf_loss(cf_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    valid = (label == 2)
    if valid.sum() < 1:
        return cf_logits.new_zeros(())
    return F.cross_entropy(cf_logits[valid], label[valid])


def info_nce(q: torch.Tensor, k: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.t() / temp
    target = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, target)


def sinkhorn_divergence(a: torch.Tensor, b: torch.Tensor, eps: float = 0.05, iters: int = 16) -> torch.Tensor:
    def _ot(x, y):
        cost = torch.cdist(x, y) ** 2
        k = torch.exp(-cost / eps)
        u = torch.ones(x.shape[0], device=x.device)
        for _ in range(iters):
            u = 1.0 / (k @ (1.0 / (k.t() @ u + 1e-9)) + 1e-9)
        v = 1.0 / (k.t() @ u + 1e-9)
        return (torch.diag(u) @ k @ torch.diag(v) * cost).sum()
    return _ot(a, b) - 0.5 * _ot(a, a) - 0.5 * _ot(b, b)


def spectral_smoothness(feat: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.fftn(feat.float(), dim=tuple(range(2, feat.dim()))).abs()
    return spec.mean()


def manifold_smoothness(feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(-1, keepdim=True).clamp_min(1e-9)
    lap = torch.diag_embed(deg.squeeze(-1)) - adj
    return torch.einsum("...nm,...nc,...mc->...", lap, feat, feat).mean()


def energy_calibration(alpha: torch.Tensor, label: torch.Tensor, margin: float = 5.0) -> torch.Tensor:
    s = alpha.sum(1)
    free_energy = -torch.logsumexp(torch.log(alpha + 1e-9), dim=1)
    target = torch.where(label == 2, s.new_full(s.shape, margin), s.new_zeros(s.shape))
    return F.smooth_l1_loss(free_energy, -target)


def compute_total(out: Dict, batch: Dict, active: Set[str], cfg: Config, epoch: int):
    w = cfg.loss
    anneal = min(1.0, (epoch + 1) / max(1, cfg.train.edl_anneal_epochs))
    terms: Dict[str, torch.Tensor] = {}
    terms["cls"] = w.cls * edl_loss(out["alpha"], batch["label"], anneal)

    if "pcad" in active and "pcad" in out:
        terms["align"] = w.align * out["pcad"]["loss_align"]
        terms["distill"] = w.distill * out["pcad"]["loss_distill"]
    if "uepd" in active and "uepd" in out:
        terms["proto"] = w.proto * out["uepd"]["loss_proto"]
        terms["ortho"] = w.ortho * out["uepd"]["loss_ortho"]
        terms["cf"] = w.cf * cf_loss(out["uepd"]["cf_logits"], batch["label"])
    if "seg" in active and "seg_logits" in out:
        terms["seg"] = w.seg * seg_loss(out["seg_logits"], batch["lesion_nc"], batch["has_lesion"])
        terms["size"] = w.size * size_loss(out["size_pred"], batch["dmax_cm"], batch["has_lesion"])
        terms["consistency"] = w.consistency * consistency_loss(out["seg_logits"], out["size_pred"], batch, cfg)
    if "afdg" in active and "afdg" in out:
        terms["freq_cons"] = w.freq_cons * out["afdg"]["loss_freq_cons"]
        terms["spatial"] = w.spatial * out["afdg"]["loss_spatial"]

    if "pcad" in active and "pcad" in out:
        for k in ("loss_bridge", "loss_mi", "loss_spectral"):
            if k in out["pcad"]:
                terms[k[5:]] = out["pcad"][k]
    if "uepd" in active and "uepd" in out:
        for k in ("loss_hyper", "loss_routing", "loss_dirichlet"):
            if k in out["uepd"]:
                terms[k[5:]] = (0.001 if k == "loss_dirichlet" else 1.0) * out["uepd"][k]
    if "afdg" in active and "afdg" in out:
        for k in ("loss_scatter", "loss_harmonic"):
            if k in out["afdg"]:
                terms[k[5:]] = out["afdg"][k]
    if "relational" in out:
        terms["relational"] = 0.1 * F.cross_entropy(out["relational"]["logits_relational"], batch["label"].clamp(min=0))
    if "moe" in out:
        terms["moe_balance"] = 0.01 * out["moe"]["loss_balance"]

    total = sum(terms.values())
    return total, terms
