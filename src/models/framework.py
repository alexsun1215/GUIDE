from typing import Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .backbone import SharedBackbone
from .heads import BaseClsHead, SizeHead, MixtureOfExpertsHead
from .pcad import PCAD
from .uepd import UEPD
from .afdg import AFDG


class RelationalReasoning(nn.Module):
    def __init__(self, channels: int, heads: int = 4, dim: int = 128, depth: int = 2, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Conv3d(channels, dim, 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1, dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                       activation="gelu", norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.proj(feat).flatten(2).transpose(1, 2) + self.pos
        x = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        for layer in self.layers:
            x = layer(x)
        emb = self.norm(x[:, 0])
        return {"embedding": emb, "logits_relational": self.head(emb)}


class GUIDE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        nc = cfg.model.num_classes
        self.backbone = SharedBackbone(cfg.model)
        self.cls_head = BaseClsHead(self.backbone.c_bottleneck, nc)
        self.size_head = SizeHead(self.backbone.c_bottleneck)
        self.pcad = PCAD(cfg.pcad, self.backbone.c_align)
        self.uepd = UEPD(cfg.uepd, self.backbone.c_bottleneck, nc)
        self.afdg = AFDG(cfg.afdg, self.backbone.c_align, nc)
        self.num_classes = nc
        if cfg.model.use_relational_reasoning:
            self.relational = RelationalReasoning(self.backbone.c_bottleneck, cfg.model.relation_heads,
                                                  cfg.model.relation_dim, cfg.model.relation_depth, nc)
        if cfg.model.use_moe_evidence:
            self.moe_head = MixtureOfExpertsHead(self.backbone.c_bottleneck, nc, cfg.model.moe_experts)

    def forward(self, batch: Dict[str, torch.Tensor], active: Set[str], training: bool = True) -> Dict:
        nc_img = batch["nc"]
        pv_img = batch.get("pv")
        need_nc_seg = "seg" in active
        feats_nc = self.backbone(nc_img, need_seg=need_nc_seg)
        feat_align_nc = feats_nc["feat_align"]
        bott_nc = feats_nc["bottleneck"]

        out: Dict = {}
        attn_bottle = None
        if "pcad" in active:
            if pv_img is None:
                raise KeyError("PCAD is active but batch has no paired 'pv' volume")
            feats_pv = self.backbone(pv_img, need_seg=False)
            pcad_out = self.pcad(feat_align_nc, feats_pv["feat_align"], batch.get("anchor"), batch.get("has_organ"))
            out["pcad"] = pcad_out
            attn_bottle = F.interpolate(pcad_out["gain_attn"], size=bott_nc.shape[2:], mode="trilinear", align_corners=False)

        base_evidence = self.cls_head(bott_nc, attn_bottle)
        evidence = base_evidence
        if "uepd" in active:
            uepd_out = self.uepd(bott_nc, batch["label"], attn_bottle)
            evidence = base_evidence + (uepd_out["alpha"] - 1.0)
            out["uepd"] = uepd_out

        alpha = evidence + 1.0
        S = alpha.sum(-1, keepdim=True)
        out["alpha"] = alpha
        out["prob"] = alpha / S
        out["uncertainty"] = self.num_classes / S.squeeze(-1)

        if "seg" in active:
            out["seg_logits"] = feats_nc["seg_logits"]
            out["size_pred"] = self.size_head(bott_nc, attn_bottle)

        if "afdg" in active:
            out["afdg"] = self.afdg(feat_align_nc, batch["center_id"], batch.get("organ_nc"), batch.get("has_organ"))

        if self.cfg.model.use_relational_reasoning:
            out["relational"] = self.relational(bott_nc)
        if self.cfg.model.use_moe_evidence:
            out["moe"] = self.moe_head(bott_nc, attn_bottle)

        return out
