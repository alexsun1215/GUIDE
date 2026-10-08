from typing import Dict, List

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    out: Dict[str, float] = {"acc": float((pred == labels).mean())}
    if _HAS_SK and len(np.unique(labels)) > 1:
        try:
            out["macro_auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except ValueError:
            out["macro_auroc"] = float("nan")
        out["macro_f1"] = float(f1_score(labels, pred, average="macro"))
        cm = confusion_matrix(labels, pred, labels=[0, 1, 2])
        tp = cm[2, 2]
        fn = cm[2, :].sum() - tp
        out["gbc_sensitivity"] = float(tp / (tp + fn + 1e-6))
        neg = cm[:2, :].sum()
        fp = cm[:2, 2].sum()
        out["gbc_specificity"] = float((neg - fp) / (neg + 1e-6))
    return out


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(labels)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def dice_score(pred_mask: np.ndarray, target: np.ndarray) -> float:
    inter = (pred_mask * target).sum()
    denom = pred_mask.sum() + target.sum()
    return float((2 * inter + 1.0) / (denom + 1.0))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def predictive_entropy(probs: np.ndarray) -> float:
    return float((-(probs * np.log(probs + 1e-9)).sum(1)).mean())


def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    acc = (probs.argmax(1) == labels).astype(np.float32)
    order = np.argsort(conf)
    e, n = 0.0, len(labels)
    for b in np.array_split(order, n_bins):
        if len(b):
            e += len(b) / n * abs(acc[b].mean() - conf[b].mean())
    return float(e)


def risk_coverage_auc(probs: np.ndarray, labels: np.ndarray, unc: np.ndarray) -> float:
    order = np.argsort(unc)
    correct = (probs.argmax(1) == labels).astype(np.float32)[order]
    cov = np.arange(1, len(labels) + 1) / len(labels)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(labels) + 1)
    return float(np.trapz(risk, cov))


def matthews_3class(probs: np.ndarray, labels: np.ndarray) -> float:
    pred = probs.argmax(1)
    k = probs.shape[1]
    cm = np.zeros((k, k))
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    t_sum, p_sum = cm.sum(1), cm.sum(0)
    s, c = cm.sum(), np.trace(cm)
    num = c * s - (t_sum * p_sum).sum()
    den = np.sqrt((s ** 2 - (p_sum ** 2).sum()) * (s ** 2 - (t_sum ** 2).sum())) + 1e-9
    return float(num / den)


def bald_score(prob_samples: np.ndarray) -> float:
    mean = prob_samples.mean(0)
    ent_mean = -(mean * np.log(mean + 1e-9)).sum(-1)
    mean_ent = -(prob_samples * np.log(prob_samples + 1e-9)).sum(-1).mean(0)
    return float((ent_mean - mean_ent).mean())


class Accumulator:
    def __init__(self):
        self.probs: List[np.ndarray] = []
        self.labels: List[int] = []
        self.unc: List[float] = []
        self.dices: List[float] = []
        self.size_err: List[float] = []

    def add_cls(self, prob: np.ndarray, label: np.ndarray, unc: np.ndarray):
        self.probs.append(prob)
        self.labels.extend(label.tolist())
        self.unc.extend(unc.tolist())

    def add_seg(self, pred_mask: np.ndarray, target: np.ndarray):
        self.dices.append(dice_score(pred_mask, target))

    def add_size(self, pred_cm: float, gt_cm: float):
        self.size_err.append(abs(pred_cm - gt_cm) * 10.0)

    def summary(self) -> Dict[str, float]:
        probs = np.concatenate(self.probs, 0)
        labels = np.array(self.labels)
        res = classification_metrics(probs, labels)
        res["ece"] = ece(probs, labels)
        if self.dices:
            res["dice"] = float(np.mean(self.dices))
        if self.size_err:
            res["size_mae_mm"] = float(np.mean(self.size_err))
        return res
