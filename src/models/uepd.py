from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import UEPDConfig
from ..utils import (expmap0, poincare_distance, sinkhorn_plan,
                     knn_adjacency, normalized_laplacian, chebyshev_basis)


class HyperbolicPrototypeHead(nn.Module):
    def __init__(self, dim: int, n_proto: int, num_classes: int, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.protos = nn.Parameter(torch.randn(n_proto, dim) * 0.02)
        self.to_class = nn.Parameter(torch.zeros(n_proto, num_classes))
        nn.init.orthogonal_(self.to_class)

    def forward(self, tokens_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = expmap0(self.scale * tokens_norm, self.c)
        p = expmap0(self.protos, self.c)
        dist = poincare_distance(x.unsqueeze(2), p.view(1, 1, *p.shape), self.c)
        logits = (-dist) @ F.softplus(self.to_class)
        evidence = F.softplus(logits).mean(1)
        compact = dist.min(-1).values.mean()
        return {"evidence": evidence, "loss_hyper": compact}


class OptimalPrototypeRouting(nn.Module):
    def __init__(self, eps: float = 0.05, iters: int = 12):
        super().__init__()
        self.eps = eps
        self.iters = iters

    def forward(self, tokens_norm: torch.Tensor, protos_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        cost = 1.0 - tokens_norm @ protos_norm.t()
        plan = sinkhorn_plan(cost.unsqueeze(0) if cost.dim() == 2 else cost, self.eps, self.iters)
        plan = plan.squeeze(0) if cost.dim() == 2 else plan
        ent = -(plan.clamp_min(1e-9) * plan.clamp_min(1e-9).log()).flatten(1).sum(-1).mean()
        return {"plan": plan, "loss_routing": (plan * cost).flatten(1).sum(-1).mean() + 0.01 * ent}


class GeodesicPropagation(nn.Module):
    def __init__(self, dim: int, k: int = 8, order: int = 3, steps: int = 2):
        super().__init__()
        self.k = k
        self.order = order
        self.steps = steps
        self.theta = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(order)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = tokens
        lap = normalized_laplacian(knn_adjacency(F.normalize(h, dim=-1), self.k))
        energy = h.new_zeros(())
        for _ in range(self.steps):
            basis = chebyshev_basis(lap, h, self.order)
            agg = sum(self.theta[i](basis[i]) for i in range(self.order))
            h = self.norm(h + F.gelu(agg))
            energy = energy + torch.einsum("...nm,...nc,...mc->...", lap, h, h).mean()
        return {"tokens": h, "loss_dirichlet": energy / max(1, self.steps)}


class UEPD(nn.Module):
    """Uncertainty-aware Exclusive Prototype Disentanglement with evidential classification."""

    def __init__(self, cfg: UEPDConfig, feat_channels: int, num_classes: int = 3):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.dim = cfg.proto_dim
        self.n_bg = cfg.n_proto_bg
        self.n_shared = cfg.n_proto_shared
        self.n_excl = cfg.n_proto_excl
        self.M = self.n_bg + self.n_shared + self.n_excl
        self.proj = nn.Sequential(
            nn.Conv3d(feat_channels, self.dim, 1), nn.GroupNorm(8, self.dim), nn.GELU(),
            nn.Conv3d(self.dim, self.dim, 1),
        )
        self.prototypes = nn.Parameter(torch.randn(self.M, self.dim) * 0.02)
        self.proto_to_class = nn.Parameter(torch.zeros(self.M, num_classes))
        with torch.no_grad():
            self.proto_to_class[: self.n_bg, 0] = 1.0
            self.proto_to_class[self.n_bg: self.n_bg + self.n_shared, 1] = 0.8
            self.proto_to_class[self.n_bg: self.n_bg + self.n_shared, 2] = 0.4
            self.proto_to_class[self.n_bg + self.n_shared:, 2] = 1.0
        self.temp = 0.1
        self.register_buffer("tau_h", torch.tensor(cfg.tau_h))
        self.register_buffer("tau_m", torch.tensor(cfg.tau_m))
        self.register_buffer("tau_u", torch.tensor(cfg.tau_u))
        if cfg.use_hyperbolic:
            self.hyper_head = HyperbolicPrototypeHead(self.dim, self.M, num_classes, cfg.curvature)
        if cfg.use_optimal_routing:
            self.router = OptimalPrototypeRouting(cfg.routing_eps, cfg.routing_iters)
        if cfg.use_geodesic_propagation:
            self.geodesic = GeodesicPropagation(self.dim, cfg.graph_k, cfg.cheb_order, cfg.propagation_steps)

    def _slices(self):
        a = self.n_bg
        b = self.n_bg + self.n_shared
        return slice(0, a), slice(a, b), slice(b, self.M)

    def _aggregate(self, feats_norm: torch.Tensor):
        P = F.normalize(self.prototypes, dim=1)
        sim = feats_norm @ P.t()
        s = F.softmax(sim / self.temp, dim=-1)
        patch_logits = (s * sim) @ self.proto_to_class
        patch_evidence = F.softplus(patch_logits)
        return sim, s, patch_evidence

    def forward(self, feat: torch.Tensor, labels: torch.Tensor, attn=None) -> Dict[str, torch.Tensor]:
        if attn is not None:
            feat = feat * (1.0 + attn)
        proj = self.proj(feat)
        b, dpc = proj.shape[0], proj.shape[1]
        tokens = proj.flatten(2).transpose(1, 2)
        fn = F.normalize(tokens, dim=-1)

        sim, s, patch_evidence = self._aggregate(fn)
        patch_alpha = patch_evidence + 1.0
        Sp = patch_alpha.sum(-1, keepdim=True)
        patch_prob = patch_alpha / Sp
        u_patch = self.num_classes / Sp.squeeze(-1)
        maxp, ypred = patch_prob.max(-1)

        img_evidence = patch_evidence.mean(1)
        alpha = img_evidence + 1.0
        S = alpha.sum(-1, keepdim=True)
        prob = alpha / S
        uncertainty = self.num_classes / S.squeeze(-1)

        lab = labels.view(-1, 1)
        cond_E = (maxp > self.tau_h) & (u_patch < self.tau_u) & (ypred == 2) & (lab == 2)
        cond_S = (maxp > self.tau_m) & (maxp <= self.tau_h) & (~cond_E)
        bank = torch.full_like(ypred, 0)
        bank[cond_S] = 1
        bank[cond_E] = 2
        bank[(maxp <= self.tau_m) | (lab == 0)] = 0

        sl_b, sl_s, sl_e = self._slices()
        best_b = sim[:, :, sl_b].max(-1).values
        best_s = sim[:, :, sl_s].max(-1).values
        best_e = sim[:, :, sl_e].max(-1).values
        best = torch.where(bank == 0, best_b, torch.where(bank == 1, best_s, best_e))
        loss_proto = (1.0 - best).mean()

        P = F.normalize(self.prototypes, dim=1)
        Ps = P[sl_s]
        Pe = P[sl_e]
        loss_ortho = (Ps @ Pe.t()).pow(2).sum()

        cf_logits = self._counterfactual(fn)

        out = {
            "alpha": alpha,
            "prob": prob,
            "uncertainty": uncertainty,
            "patch_prob": patch_prob,
            "u_patch": u_patch,
            "cf_logits": cf_logits,
            "loss_proto": loss_proto,
            "loss_ortho": loss_ortho,
            "bank": bank,
        }
        if self.cfg.use_hyperbolic:
            hy = self.hyper_head(fn)
            out["hyper_evidence"] = hy["evidence"]
            out["loss_hyper"] = self.cfg.lambda_hyper * hy["loss_hyper"]
        if self.cfg.use_optimal_routing:
            rt = self.router(fn.reshape(-1, self.dim), P)
            out["loss_routing"] = rt["loss_routing"]
        if self.cfg.use_geodesic_propagation:
            gp = self.geodesic(tokens)
            out["geodesic_tokens"] = gp["tokens"]
            out["loss_dirichlet"] = gp["loss_dirichlet"]
        return out

    def _counterfactual(self, fn: torch.Tensor) -> torch.Tensor:
        _, sl_s, _ = self._slices()
        with torch.no_grad():
            Ps = F.normalize(self.prototypes[sl_s], dim=1)
            q, _ = torch.linalg.qr(Ps.t().float().cpu())
            q = q.to(fn.device, fn.dtype)
        proj = fn @ q @ q.t()
        f_tilde = F.normalize(fn - proj, dim=-1)
        _, _, ev = self._aggregate(f_tilde)
        cf_alpha = ev.mean(1) + 1.0
        return torch.log(cf_alpha)

    @torch.no_grad()
    def update_thresholds(self, maxp: torch.Tensor, u: torch.Tensor):
        d = self.cfg.thresh_ema
        self.tau_h.mul_(d).add_((1 - d) * maxp.float().quantile(0.7).clamp(0.7, 0.9))
        self.tau_m.mul_(d).add_((1 - d) * maxp.float().quantile(0.4).clamp(0.4, 0.6))
        self.tau_u.mul_(d).add_((1 - d) * u.float().quantile(0.5).clamp(0.2, 0.4))
