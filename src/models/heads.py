import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfExpertsHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 3, experts: int = 4, hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.gate = nn.Linear(in_channels, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
            for _ in range(experts)
        ])

    def forward(self, feat: torch.Tensor, attn: torch.Tensor = None):
        if attn is not None:
            feat = feat * (1.0 + attn)
        z = self.pool(feat).flatten(1)
        g = F.softmax(self.gate(z), dim=-1)
        ev = torch.stack([F.softplus(e(z)) for e in self.experts], dim=1)
        load = g.mean(0)
        balance = (load * load.log().clamp_min(-20)).sum() + torch.log(torch.tensor(float(g.shape[1])))
        return {"evidence": (g.unsqueeze(-1) * ev).sum(1), "gate": g, "loss_balance": balance}


class BaseClsHead(nn.Module):
    """Evidential classification head producing non-negative evidence."""

    def __init__(self, in_channels: int, num_classes: int = 3, hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(in_channels, hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden, num_classes),
        )

    def forward(self, feat: torch.Tensor, attn: torch.Tensor = None) -> torch.Tensor:
        if attn is not None:
            feat = feat * (1.0 + attn)
        logits = self.mlp(self.pool(feat))
        return torch.nn.functional.softplus(logits)


class SizeHead(nn.Module):
    """Max-diameter regression head (cm)."""

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(in_channels, hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor, attn: torch.Tensor = None) -> torch.Tensor:
        if attn is not None:
            feat = feat * (1.0 + attn)
        return torch.nn.functional.softplus(self.mlp(self.pool(feat))).squeeze(-1)
