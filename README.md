# 🩻 GUIDE

**Gallbladder Cancer Unenhanced CT Intelligent Detection and Evaluation**

GUIDE is a deep learning system for gallbladder cancer (GBC) detection and lesion characterization on unenhanced CT (NC).

GUIDE uses paired NC and portal venous-phase contrast-enhanced CT (PV) during training and **requires NC alone at inference**. Its outputs include:

- Three-class probabilities: `0 = normal gallbladder`, `1 = benign gallbladder disease`, and `2 = GBC`.
- Classification uncertainty.
- Gallbladder and lesion probability maps.
- Maximum lesion diameter in millimeters, with lesion-size assessment at a 10-mm threshold.

## 🏥 Study overview

The retrospective study included **1533 patients from seven Chinese hospitals**: 282 normal, 510 benign, and 741 GBC cases.

| Partition | Source | Patients |
| --- | --- | ---: |
| Training | Southwest Hospital | 718 |
| Internal validation | Southwest Hospital | 90 |
| Internal test | Southwest Hospital | 89 |
| External validation | Six other hospitals | 636 |

Patients did not overlap between partitions. Only the training partition was used for parameter updates; internal validation was used for model selection and hyperparameter tuning.

Annotations were available at three levels: **L1**, case-level labels (1533 patients); **L2**, paired NC/PV gallbladder masks (449 patients); and **L3**, PV lesion masks and maximum-diameter measurements (156 patients). These counts refer to the full study cohort. The annotation levels were not strictly nested, and each supervised loss used only the annotations available for a training case.

## 🧠 Method

GUIDE first localizes the gallbladder, then analyzes the gallbladder region and surrounding tissue. A **SwinUNETR-v2** backbone combines anatomical guidance with three modules:

- **PCAD — Phase-Contrast Anatomical Distillation:** transfers information from paired PV images to the NC pathway through anatomy-guided alignment and confidence-weighted distillation.
- **UEPD — Uncertainty-aware Evidential Prototype Disentanglement:** separates background, shared benign–malignant, and cancer-specific representations using anatomical guidance and classification uncertainty.
- **AFDG — Anatomy-anchored Frequency-Domain Generalization:** exchanges high-frequency feature components between randomly paired training cases while preserving low-frequency components and encouraging consistent predictions. It does not require center labels.

## 📊 Key results

| Evaluation | Result |
| --- | ---: |
| External AUC, GBC versus all non-GBC cases (n = 636) | 0.970 |
| External AUC, GBC versus benign disease (n = 604) | 0.962 |
| External three-class accuracy (n = 636) | 86.9% |
| Lesion segmentation Dice (70 external GBC cases with L3 annotations) | 0.660 |
| Maximum-diameter mean absolute error (same 70 cases) | 3.09 mm |

In a paired reader study involving six radiologists and 160 examinations, GUIDE assistance increased the mean AUC for GBC versus non-GBC from **0.801 to 0.869**.

## 📂 Data availability

De-identified participant-level data and a data dictionary will be available upon reasonable request after publication, subject to institutional approval and a signed data use agreement. Requests should be directed to the corresponding author with a research proposal specifying the intended analyses.
