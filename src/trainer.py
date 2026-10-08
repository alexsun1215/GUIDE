import os
from typing import Dict, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .utils import get_device, set_seed, setup_logger, count_params
from .data.manifest import build_index, split_index, summarize
from .data.dataset import GBCDataset
from .data.sampler import HierarchicalBatchSampler
from .models.framework import GUIDE
from .losses import compute_total
from .metrics import Accumulator


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)


class SharpnessAwareWrapper:
    def __init__(self, params, rho: float = 0.05):
        self.params = [p for p in params if p.requires_grad]
        self.rho = rho
        self._e = []

    @torch.no_grad()
    def ascend(self):
        norm = torch.norm(torch.stack([p.grad.norm() for p in self.params if p.grad is not None]) + 1e-12)
        self._e = []
        for p in self.params:
            if p.grad is None:
                self._e.append(None)
                continue
            e = self.rho * p.grad / (norm + 1e-12)
            p.add_(e)
            self._e.append(e)

    @torch.no_grad()
    def descend(self):
        for p, e in zip(self.params, self._e):
            if e is not None:
                p.sub_(e)


class Trainer:
    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = torch.device(device) if device else get_device()
        set_seed(cfg.train.seed)
        self.logger = setup_logger(cfg.train.out_dir)

        samples = split_index(build_index(cfg.data), cfg.data, seed=cfg.train.seed)
        self.logger.info(f"data summary: {summarize(samples)}")
        self.train_ds = GBCDataset(samples, cfg.data, "train")
        self.val_ds = GBCDataset(samples, cfg.data, "val")
        self.test_ds = GBCDataset(samples, cfg.data, "test")

        sampler = HierarchicalBatchSampler(self.train_ds, cfg.train.sampler_ratio, cfg.train.seed)
        self.train_loader = DataLoader(self.train_ds, batch_sampler=sampler, num_workers=cfg.data.num_workers)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
        self.test_loader = DataLoader(self.test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

        self.model = GUIDE(cfg).to(self.device)
        self.logger.info(f"model params: {count_params(self.model):.1f}M on {self.device}")
        self.amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.global_epoch = 0
        self.ema = ModelEMA(self.model, cfg.train.ema_decay_model) if cfg.train.use_ema else None

    def _build_optimizer(self, freeze_backbone: bool):
        for p in self.model.backbone.parameters():
            p.requires_grad = not freeze_backbone
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _step(self, batch, active: Set[str], optimizer) -> Dict[str, float]:
        batch = to_device(batch, self.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=self.amp):
            out = self.model(batch, active, training=True)
            total, terms = compute_total(out, batch, active, self.cfg, self.global_epoch)
        if self.amp:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            optimizer.step()
        if "uepd" in active and "uepd" in out:
            self.model.uepd.update_thresholds(out["uepd"]["patch_prob"].max(-1).values.detach(),
                                              out["uepd"]["u_patch"].detach())
        if self.ema is not None:
            self.ema.update(self.model)
        return {k: float(v.detach()) for k, v in terms.items()}

    def fit(self):
        for stage in self.cfg.train.stages:
            active = set(stage.modules)
            optimizer = self._build_optimizer(stage.freeze_backbone)
            self.logger.info(f"=== {stage.name} active={sorted(active)} freeze_bb={stage.freeze_backbone} ===")
            for _ in range(stage.epochs):
                self.model.train()
                agg: Dict[str, float] = {}
                for batch in self.train_loader:
                    terms = self._step(batch, active, optimizer)
                    for k, v in terms.items():
                        agg[k] = agg.get(k, 0.0) + v
                n = max(1, len(self.train_loader))
                msg = " ".join(f"{k}={v / n:.4f}" for k, v in agg.items())
                self.logger.info(f"[{stage.name}] epoch {self.global_epoch} {msg}")
                self.global_epoch += 1
            metrics = self.evaluate(active)
            self.logger.info(f"[{stage.name}] val {metrics}")
            self._save(stage.name)
        return self.test()

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Independent evaluation on the held-out external centers (NC-only deploy path)."""
        deploy_active = {"cls", "uepd", "seg"}
        metrics = self.evaluate(deploy_active, loader=self.test_loader)
        if metrics:
            self.logger.info(f"[external_test] NC-only deploy metrics {metrics}")
        else:
            self.logger.info("[external_test] no external-center samples to evaluate")
        return metrics

    @torch.no_grad()
    def evaluate(self, active: Set[str], loader=None) -> Dict[str, float]:
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        acc = Accumulator()
        eval_active = set(active)
        for batch in loader:
            batch = to_device(batch, self.device)
            out = self.model(batch, eval_active, training=False)
            prob = out["prob"].float().cpu().numpy()
            acc.add_cls(prob, batch["label"].cpu().numpy(), out["uncertainty"].float().cpu().numpy())
            if "seg_logits" in out:
                pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float().cpu().numpy()
                tgt = batch["lesion_nc"].cpu().numpy()
                hl = batch["has_lesion"].cpu().numpy()
                size_pred = out["size_pred"].float().cpu().numpy()
                dmax = batch["dmax_cm"].cpu().numpy()
                for i in range(pred.shape[0]):
                    if hl[i] > 0.5:
                        acc.add_seg(pred[i, 0], tgt[i, 0])
                        if dmax[i] > 0:
                            acc.add_size(float(size_pred[i]), float(dmax[i]))
        if not acc.probs:
            return {}
        return acc.summary()

    def _save(self, name: str):
        path = os.path.join(self.cfg.train.out_dir, f"{name}.pt")
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, path)
