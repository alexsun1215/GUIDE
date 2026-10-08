import os
import hashlib
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, map_coordinates

from ..config import DataConfig
from .manifest import Sample

try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False


def _seed_of(pid: str) -> int:
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % (2 ** 31)


def _resize(vol: np.ndarray, size, mode: str) -> np.ndarray:
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = torch.nn.functional.interpolate(t, size=size, mode=mode, align_corners=False if mode == "trilinear" else None)
    return t[0, 0].numpy()


def _load_nifti(path: str, size, is_mask: bool, clip_hu) -> np.ndarray:
    vol = nib.load(path).get_fdata().astype(np.float32)
    if is_mask:
        vol = (vol > 0.5).astype(np.float32)
        vol = _resize(vol, size, "nearest")
        return (vol > 0.5).astype(np.float32)
    vol = np.clip(vol, clip_hu[0], clip_hu[1])
    vol = (vol - clip_hu[0]) / (clip_hu[1] - clip_hu[0] + 1e-6)
    vol = _resize(vol, size, "trilinear")
    return vol


def _ellipsoid(size, center, radii) -> np.ndarray:
    d, h, w = size
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    val = ((zz - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2 + ((xx - center[2]) / radii[2]) ** 2
    return (val <= 1.0).astype(np.float32)


def _warp(vol: np.ndarray, disp: np.ndarray, order: int) -> np.ndarray:
    d, h, w = vol.shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([zz + disp[0], yy + disp[1], xx + disp[2]], axis=0)
    return map_coordinates(vol, coords, order=order, mode="nearest").astype(np.float32)


def _synth_scene(sample: Sample, size, clip_hu) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(_seed_of(sample.pid))
    d, h, w = size
    nc = rng.normal(0.25, 0.03, size).astype(np.float32)
    center = np.array([d / 2, h / 2, w / 2]) + rng.uniform(-d * 0.08, d * 0.08, 3)
    radii = np.array([d, h, w]) * rng.uniform(0.16, 0.22)
    organ = _ellipsoid(size, center, radii)
    wall = organ - _ellipsoid(size, center, radii * 0.78)
    wall = np.clip(wall, 0, 1)
    nc += organ * 0.18 + wall * 0.05
    pv_base = nc.copy()

    lesion = np.zeros(size, np.float32)
    if sample.label >= 1:
        ang = rng.uniform(0, 2 * np.pi)
        lpos = center + np.array([rng.uniform(-2, 2), np.sin(ang), np.cos(ang)]) * radii * 0.8
        if sample.has_lesion and sample.dmax_mm > 0:
            lr = max(2.0, (sample.dmax_mm / 10.0) / 6.0 * d * 0.16)
        else:
            lr = d * rng.uniform(0.04, 0.09)
        lesion = _ellipsoid(size, lpos, np.array([lr, lr, lr]))
        enh = 0.30 if sample.label == 2 else 0.16
        enh *= rng.uniform(0.85, 1.15)
        pv_base += lesion * enh
        nc += lesion * (enh * 0.25)
        nc += wall * (0.04 if sample.label == 2 else 0.06)

    cid = _seed_of(sample.hospital) % 7
    freq = 2 + cid
    zz, yy, xx = np.meshgrid(np.linspace(0, np.pi, d), np.linspace(0, np.pi, h), np.linspace(0, np.pi, w), indexing="ij")
    style = 0.02 * (cid / 6.0 + 0.3) * np.sin(freq * zz) * np.cos(freq * yy) * np.sin(freq * xx)
    nc = nc + style.astype(np.float32)
    pv_base = pv_base + style.astype(np.float32)

    base_disp = rng.uniform(-2.5, 2.5, 3)
    field = np.stack([gaussian_filter(rng.uniform(-1, 1, size), 4) for _ in range(3)], axis=0)
    disp = base_disp[:, None, None, None] + field * 1.5
    pv = _warp(pv_base, disp, order=1)
    organ_pv = (_warp(organ, disp, order=0) > 0.5).astype(np.float32)
    lesion_pv = (_warp(lesion, disp, order=0) > 0.5).astype(np.float32)

    nc = np.clip(nc, 0, 1)
    pv = np.clip(pv, 0, 1)
    return {
        "nc": nc, "pv": pv,
        "organ_nc": organ, "organ_pv": organ_pv,
        "lesion_nc": lesion, "lesion_pv": lesion_pv,
    }


def _anchor_map(organ: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(organ)[None, None]
    eroded = -torch.nn.functional.max_pool3d(-t, kernel_size=3, stride=1, padding=1)
    boundary = (t - eroded)[0, 0].clamp(0, 1).numpy()
    return boundary.astype(np.float32)


class FourierDomainMixing:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def __call__(self, vol: np.ndarray, ref: np.ndarray) -> np.ndarray:
        fa = np.fft.fftn(vol)
        ab = np.abs(np.fft.fftn(ref))
        lam = np.random.uniform(0, self.alpha)
        mixed = (1 - lam) * np.abs(fa) + lam * ab
        return np.fft.ifftn(mixed * np.exp(1j * np.angle(fa))).real.astype(np.float32)


class ElasticDeformation3D:
    def __init__(self, sigma: float = 4.0, alpha: float = 8.0):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        field = [gaussian_filter(np.random.uniform(-1, 1, vol.shape), self.sigma) * self.alpha for _ in range(3)]
        grids = np.meshgrid(*[np.arange(s) for s in vol.shape], indexing="ij")
        coords = np.stack([g + f for g, f in zip(grids, field)], axis=0)
        return map_coordinates(vol, coords, order=1, mode="nearest").astype(np.float32)


class RandomGammaJitter:
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        g = np.random.uniform(*self.gamma_range)
        return (np.clip(vol, 0, 1) ** g).astype(np.float32)


class CutMix3D:
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lam = np.random.beta(self.beta, self.beta)
        dims = a.shape
        rs = [int(s * (1 - lam) ** (1.0 / 3.0)) for s in dims]
        st = [np.random.randint(0, max(1, s - r + 1)) for s, r in zip(dims, rs)]
        out = a.copy()
        sl = tuple(slice(s, s + r) for s, r in zip(st, rs))
        out[sl] = b[sl]
        return out.astype(np.float32)


class GBCDataset(Dataset):
    def __init__(self, samples: List[Sample], cfg: DataConfig, split: str):
        self.cfg = cfg
        self.split = split
        self.samples = [s for s in samples if s.split == split]
        self.size = tuple(cfg.img_size)
        self.centers = sorted({s.hospital for s in samples})
        self.center_to_id = {c: i for i, c in enumerate(self.centers)}

    def __len__(self):
        return len(self.samples)

    def _try_real(self, s: Sample):
        root = self.cfg.data_root
        nc_p = os.path.join(root, self.cfg.nc_pattern.format(pid=s.pid))
        pv_p = os.path.join(root, self.cfg.pv_pattern.format(pid=s.pid))
        if not (_HAS_NIB and os.path.exists(nc_p) and os.path.exists(pv_p)):
            return None
        out = {
            "nc": _load_nifti(nc_p, self.size, False, self.cfg.clip_hu),
            "pv": _load_nifti(pv_p, self.size, False, self.cfg.clip_hu),
        }
        for key, pat in (("organ_nc", self.cfg.organ_nc_pattern), ("organ_pv", self.cfg.organ_pv_pattern)):
            p = os.path.join(root, pat.format(pid=s.pid))
            out[key] = _load_nifti(p, self.size, True, self.cfg.clip_hu) if os.path.exists(p) else np.zeros(self.size, np.float32)
        lp = os.path.join(root, self.cfg.lesion_pattern.format(lesion_file=s.lesion_file or s.pid))
        lesion = _load_nifti(lp, self.size, True, self.cfg.clip_hu) if os.path.exists(lp) else np.zeros(self.size, np.float32)
        out["lesion_pv"] = lesion
        out["lesion_nc"] = lesion
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        scene = self._try_real(s)
        if scene is None:
            if not self.cfg.allow_synthetic:
                raise FileNotFoundError(f"missing volume for {s.pid} and synthetic disabled")
            scene = _synth_scene(s, self.size, self.cfg.clip_hu)

        anchor = _anchor_map(scene["organ_nc"]) if s.has_organ else np.zeros(self.size, np.float32)
        item = {
            "nc": torch.from_numpy(scene["nc"]).float().unsqueeze(0),
            "pv": torch.from_numpy(scene["pv"]).float().unsqueeze(0),
            "organ_nc": torch.from_numpy(scene["organ_nc"]).float().unsqueeze(0),
            "organ_pv": torch.from_numpy(scene["organ_pv"]).float().unsqueeze(0),
            "lesion_nc": torch.from_numpy(scene["lesion_nc"]).float().unsqueeze(0),
            "lesion_pv": torch.from_numpy(scene["lesion_pv"]).float().unsqueeze(0),
            "anchor": torch.from_numpy(anchor).float().unsqueeze(0),
            "label": torch.tensor(s.label, dtype=torch.long),
            "dmax_cm": torch.tensor(s.dmax_mm / 10.0 if s.dmax_mm > 0 else -1.0, dtype=torch.float32),
            "has_organ": torch.tensor(1.0 if s.has_organ else 0.0),
            "has_lesion": torch.tensor(1.0 if s.has_lesion else 0.0),
            "level": torch.tensor(s.level, dtype=torch.long),
            "center_id": torch.tensor(self.center_to_id.get(s.hospital, 0), dtype=torch.long),
        }
        return item
