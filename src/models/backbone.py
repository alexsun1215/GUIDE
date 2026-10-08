from typing import Dict

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from ..config import ModelConfig


class SharedBackbone(nn.Module):
    """SwinUNETR-v2 encoder/decoder exposing multi-level features and segmentation logits."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = SwinUNETR(
            in_channels=cfg.in_channels,
            out_channels=cfg.seg_channels,
            feature_size=cfg.feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            dropout_path_rate=cfg.dropout_path_rate,
            use_v2=cfg.use_v2,
            spatial_dims=3,
        )
        fs = cfg.feature_size
        self.c_align = 4 * fs
        self.c_bottleneck = 16 * fs
        self.c_dec = fs

    def forward(self, x: torch.Tensor, need_seg: bool = True) -> Dict[str, torch.Tensor]:
        net = self.net
        hidden = net.swinViT(x, net.normalize)
        enc0 = net.encoder1(x)
        enc1 = net.encoder2(hidden[0])
        enc2 = net.encoder3(hidden[1])
        enc3 = net.encoder4(hidden[2])
        dec4 = net.encoder10(hidden[4])
        out: Dict[str, torch.Tensor] = {
            "feat_align": hidden[2],
            "bottleneck": dec4,
        }
        if need_seg:
            d3 = net.decoder5(dec4, hidden[3])
            d2 = net.decoder4(d3, enc3)
            d1 = net.decoder3(d2, enc2)
            d0 = net.decoder2(d1, enc1)
            dec_feat = net.decoder1(d0, enc0)
            out["dec_feat"] = dec_feat
            out["seg_logits"] = net.out(dec_feat)
        return out
