import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from ..config import DataConfig

PATHO_TO_LABEL = {
    "健康": 0,
    "结石": 1,
    "腺肌症": 1,
    "息肉": 1,
    "慢性炎": 1,
    "GBC": 2,
}


@dataclass
class Sample:
    pid: str
    hospital: str
    label: int
    pathology: str
    has_organ: bool = False
    has_lesion: bool = False
    dmax_mm: float = -1.0
    lesion_file: Optional[str] = None
    split: str = "train"

    @property
    def level(self) -> int:
        if self.has_lesion:
            return 3
        if self.has_organ:
            return 2
        return 1


def _to_label(v) -> int:
    try:
        f = float(v)
        if f in (0.0, 1.0, 2.0):
            return int(f)
    except (TypeError, ValueError):
        pass
    return -1


def _norm_pid(v) -> str:
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def _load_lesions(cfg: DataConfig) -> Dict[str, Dict]:
    path = cfg.lesion_manifest
    out: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return out
    df = pd.read_excel(path, sheet_name=cfg.lesion_sheet, skiprows=cfg.lesion_skiprows, header=None)
    for _, row in df.iterrows():
        pid = _norm_pid(row[0])
        if not pid or pid == "nan":
            continue
        lesion_file = str(row[1]).strip() if not pd.isna(row[1]) else None
        if lesion_file is None or "对应" in lesion_file:
            continue
        try:
            dmax = float(row[4])
        except (TypeError, ValueError):
            continue
        ptype = str(row[2]).strip()
        out[pid] = {"lesion_file": lesion_file, "dmax_mm": dmax, "type": ptype}
    return out


def synthetic_index(n: int = 160, seed: int = 2026) -> List[Sample]:
    rng = np.random.RandomState(seed)
    centers = ["西南医院", "新华医院", "河南人民医院", "郑大一附院", "新桥医院"]
    samples: List[Sample] = []
    for i in range(n):
        label = int(rng.choice([0, 1, 2], p=[0.21, 0.35, 0.44]))
        hosp = centers[rng.randint(len(centers))]
        has_lesion = bool(rng.rand() < 0.18)
        has_organ = bool(rng.rand() < 0.33) or has_lesion
        dmax = float(rng.uniform(5, 30)) if has_lesion else -1.0
        samples.append(Sample(
            pid=f"SYN{i:05d}", hospital=hosp, label=label, pathology="synthetic",
            has_organ=has_organ, has_lesion=has_lesion, dmax_mm=dmax,
            lesion_file=f"SYN{i:05d}-V" if has_lesion else None,
        ))
    return samples


def build_index(cfg: DataConfig) -> List[Sample]:
    if not os.path.exists(cfg.case_manifest):
        return synthetic_index()
    df = pd.read_excel(cfg.case_manifest, sheet_name=cfg.case_sheet, skiprows=cfg.case_skiprows, header=None)
    df = df.iloc[:, :6]
    df.columns = ["pid", "hospital", "label", "patho", "pv_mask", "nc_mask"]
    lesions = _load_lesions(cfg)

    samples: List[Sample] = []
    seen = set()
    for _, row in df.iterrows():
        if pd.isna(row["pid"]):
            continue
        pid = _norm_pid(row["pid"])
        label = _to_label(row["label"])
        if label < 0:
            continue
        hospital = str(row["hospital"]).strip()
        has_organ = (str(row["pv_mask"]).strip() in ("1", "1.0")) and (str(row["nc_mask"]).strip() in ("1", "1.0"))
        s = Sample(
            pid=pid,
            hospital=hospital,
            label=label,
            pathology=str(row["patho"]).strip(),
            has_organ=has_organ,
        )
        if pid in lesions:
            s.has_lesion = True
            s.dmax_mm = float(lesions[pid]["dmax_mm"])
            s.lesion_file = lesions[pid]["lesion_file"]
        samples.append(s)
        seen.add(pid)

    for pid, info in lesions.items():
        if pid in seen:
            continue
        label = PATHO_TO_LABEL.get(info["type"].replace("（胆囊癌）", ""), -1)
        if "GBC" in info["type"]:
            label = 2
        if label < 0:
            continue
        samples.append(Sample(
            pid=pid, hospital="unknown", label=label, pathology=info["type"],
            has_organ=False, has_lesion=True, dmax_mm=float(info["dmax_mm"]),
            lesion_file=info["lesion_file"],
        ))
    return samples


def split_index(samples: List[Sample], cfg: DataConfig, val_ratio: float = 0.1, seed: int = 2026) -> List[Sample]:
    rng = np.random.RandomState(seed)
    ext = set(cfg.external_centers)
    train_pool = []
    for s in samples:
        if s.hospital in ext:
            s.split = "test"
        else:
            train_pool.append(s)
    by_label: Dict[int, List[Sample]] = {0: [], 1: [], 2: []}
    for s in train_pool:
        by_label[s.label].append(s)
    for label, group in by_label.items():
        idx = rng.permutation(len(group))
        n_val = max(1, int(len(group) * val_ratio)) if group else 0
        for j, gi in enumerate(idx):
            group[gi].split = "val" if j < n_val else "train"
    return samples


def summarize(samples: List[Sample]) -> Dict[str, int]:
    stat = {
        "total": len(samples),
        "label0": sum(s.label == 0 for s in samples),
        "label1": sum(s.label == 1 for s in samples),
        "label2": sum(s.label == 2 for s in samples),
        "L2_organ": sum(s.has_organ for s in samples),
        "L3_lesion": sum(s.has_lesion for s in samples),
        "train": sum(s.split == "train" for s in samples),
        "val": sum(s.split == "val" for s in samples),
        "test": sum(s.split == "test" for s in samples),
    }
    return stat
