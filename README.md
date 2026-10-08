# 🧭 GUIDE

**Gallbladder Cancer Unenhanced CT Intelligent Detection and Evaluation**

Gallbladder cancer detection and lesion characterization on unenhanced CT.

**🧠 Training:** paired NC + PV CT　·　**🩻 Inference:** NC CT alone

[🔎 Overview](#overview) · [🧠 Framework](#framework) · [🏷️ Supervision](#supervision) · [🧪 Study design](#study-design) · [📂 Availability](#availability)

<a id="overview"></a>

## 🔎 Overview

Gallbladder cancer can resemble benign inflammatory disease on unenhanced CT. **GUIDE** combines diagnostic classification, anatomical localization, and quantitative lesion assessment to support gallbladder evaluation from a single unenhanced examination.

Paired unenhanced CT (**NC**) and portal venous-phase contrast-enhanced CT (**PV**) provide supervision during training. **Only NC is required at inference.**

| Capability | Output |
| :--- | :--- |
| **🩺 Diagnosis** | Probabilities for normal gallbladder, benign disease, and gallbladder cancer, with classification uncertainty |
| **🧩 Localization and segmentation** | Gallbladder location and probability maps for the gallbladder and lesions |
| **📏 Lesion measurement** | Maximum lesion diameter in millimeters |

<a id="framework"></a>

## 🧠 Framework

GUIDE first localizes the gallbladder, then analyzes the gallbladder region and surrounding tissue using a **SwinUNETR-v2** backbone. Three modules support cross-phase learning, benign–malignant differentiation, and generalization across imaging conditions.

**🔄 PCAD — Phase-Contrast Anatomical Distillation**

Transfers enhancement-related information from paired PV images to the NC pathway. Anatomy-guided deformable alignment establishes feature correspondence, while confidence-weighted residual distillation guides learning from cross-phase differences.

**🎯 UEPD — Uncertainty-aware Evidential Prototype Disentanglement**

Separates background, shared benign–malignant, and cancer-specific representations. Anatomical guidance, prediction uncertainty, and prototype constraints help identify cancer evidence beyond features shared with benign disease.

**🌐 AFDG — Anatomy-anchored Frequency-Domain Generalization**

Aims to reduce reliance on acquisition-related texture by exchanging high-frequency features between randomly paired training cases while preserving low-frequency components. Prediction consistency and anatomical constraints guide learning without requiring center labels.

<a id="supervision"></a>

## 🏷️ Supervision

Each training case contributes to the tasks for which reference annotations are available.

| Level | Annotation | Role |
| :---: | :--- | :--- |
| **L1** | Case-level diagnostic label | Three-class classification |
| **L2** | Gallbladder masks on paired NC and PV images | Gallbladder segmentation and anatomical guidance |
| **L3** | PV lesion masks and maximum-diameter measurements | Lesion segmentation and diameter estimation |

The levels are **not strictly nested**; each supervised loss is activated only when its required annotations are available. PV-derived lesion masks are transferred to NC space for lesion supervision.

<a id="study-design"></a>

## 🧪 Study design

The retrospective multicenter study uses independent patient partitions for model development and evaluation.

| Component | Design |
| :--- | :--- |
| **⚙️ Development** | Training and internal validation at one center, with internal validation used for model selection and hyperparameter tuning |
| **🧪 Internal testing** | A separate cohort from the development center, held out from model fitting and selection |
| **🌍 External validation** | Independent evaluation at six additional centers excluded from model development |
| **👥 Reader study** | Six radiologists interpret 160 held-out examinations without and with GUIDE assistance, separated by a one-month washout |

Patients do not overlap between partitions. Evaluation covers three-class diagnosis, GBC versus non-GBC and versus benign disease, lesion segmentation, maximum-diameter estimation, and reader assistance. Lesion-size analyses use a **10-mm threshold**.

<a id="availability"></a>

## 📂 Availability

**💻 Code.** Source code is available in this repository under the [MIT licence](https://github.com/alexsun1215/GUIDE/blob/main/LICENSE). The associated manuscript is currently under review.

**🔒 Data.** Clinical images and participant-level datasets are not included in this repository. After publication, de-identified participant-level data and a data dictionary will be available upon reasonable request to the corresponding author, subject to institutional approval and a signed data use agreement.
