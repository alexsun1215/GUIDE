from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class DataConfig:
    case_manifest: str = "../数据汇总4.25.xlsx"
    case_sheet: str = "Sheet1"
    case_skiprows: int = 2
    lesion_manifest: str = "../LesionAnnotation-427.xlsx"
    lesion_sheet: str = "标注记录表"
    lesion_skiprows: int = 1
    data_root: str = "./data_root"
    nc_pattern: str = "nc/{pid}.nii.gz"
    pv_pattern: str = "pv/{pid}.nii.gz"
    organ_nc_pattern: str = "organ_nc/{pid}.nii.gz"
    organ_pv_pattern: str = "organ_pv/{pid}.nii.gz"
    lesion_pattern: str = "lesion/{lesion_file}.nii.gz"
    img_size: Tuple[int, int, int] = (96, 96, 96)
    clip_hu: Tuple[float, float] = (-100.0, 250.0)
    voxel_size_cm: float = 0.15
    allow_synthetic: bool = True
    num_workers: int = 4
    external_centers: Tuple[str, ...] = ("新华医院", "河南人民医院", "郑大一附院")


@dataclass
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 3
    seg_channels: int = 1
    feature_size: int = 48
    use_v2: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.1
    use_relational_reasoning: bool = False
    relation_heads: int = 4
    relation_dim: int = 128
    relation_depth: int = 2
    use_moe_evidence: bool = False
    moe_experts: int = 4


@dataclass
class PCADConfig:
    deform_hidden: int = 64
    lambda_smooth: float = 1.0
    lambda_anchor: float = 2.0
    anchor_weight: float = 5.0
    gate_alpha_smooth: float = 1.0
    gate_alpha_residual: float = 1.0
    gate_alpha_jac: float = 1.0
    anchor_tau: float = 2.0
    use_sinkhorn_bridge: bool = False
    bridge_tokens: int = 6
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 16
    lambda_bridge: float = 0.2
    use_cross_phase_mi: bool = False
    mi_hidden: int = 128
    lambda_mi: float = 0.1
    use_spectral_consistency: bool = False
    spectral_bands: int = 6
    lambda_spectral: float = 0.05


@dataclass
class UEPDConfig:
    n_proto_bg: int = 4
    n_proto_shared: int = 6
    n_proto_excl: int = 6
    proto_dim: int = 128
    ema_decay: float = 0.99
    tau_h: float = 0.8
    tau_m: float = 0.5
    tau_u: float = 0.3
    thresh_ema: float = 0.98
    use_hyperbolic: bool = False
    curvature: float = 1.0
    lambda_hyper: float = 0.1
    use_optimal_routing: bool = False
    routing_eps: float = 0.05
    routing_iters: int = 12
    use_geodesic_propagation: bool = False
    graph_k: int = 8
    cheb_order: int = 3
    propagation_steps: int = 2


@dataclass
class AFDGConfig:
    cutoff_ratio: float = 0.25
    lambda_spatial: float = 1.0
    use_wavelet_scattering: bool = False
    scattering_scales: int = 3
    scattering_order: int = 2
    lambda_scatter: float = 0.05
    use_harmonic_descriptors: bool = False
    n_harmonics: int = 9
    lambda_harmonic: float = 0.05


@dataclass
class LossWeights:
    cls: float = 1.0
    align: float = 0.5
    distill: float = 1.0
    proto: float = 0.5
    cf: float = 0.3
    ortho: float = 0.1
    seg: float = 1.0
    size: float = 0.5
    consistency: float = 0.2
    freq_cons: float = 0.3
    spatial: float = 0.1


@dataclass
class StageConfig:
    name: str
    epochs: int
    modules: Tuple[str, ...]
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    grad_clip: float = 5.0
    amp: bool = True
    seed: int = 2026
    out_dir: str = "./runs/guide_v31"
    sampler_ratio: Tuple[int, int, int] = (4, 2, 2)
    edl_anneal_epochs: int = 30
    use_ema: bool = False
    ema_decay_model: float = 0.999
    use_sam: bool = False
    sam_rho: float = 0.05
    use_stochastic_weight_avg: bool = False
    swa_start_frac: float = 0.75
    stages: Tuple[StageConfig, ...] = (
        StageConfig("stage1_cls", 30, ("cls",)),
        StageConfig("stage2_pcad", 20, ("cls", "pcad"), freeze_backbone=True),
        StageConfig("stage3_uepd", 10, ("cls", "pcad", "uepd"), freeze_backbone=True),
        StageConfig("stage4_seg", 30, ("cls", "pcad", "uepd", "seg")),
        StageConfig("stage5_joint", 20, ("cls", "pcad", "uepd", "seg", "afdg")),
        StageConfig("stage6_afdg", 10, ("cls", "afdg"), freeze_backbone=True),
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pcad: PCADConfig = field(default_factory=PCADConfig)
    uepd: UEPDConfig = field(default_factory=UEPDConfig)
    afdg: AFDGConfig = field(default_factory=AFDGConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)


def smoke_config() -> Config:
    cfg = Config()
    cfg.data.img_size = (64, 64, 64)
    cfg.data.allow_synthetic = True
    cfg.data.case_manifest = "./__no_manifest__.xlsx"
    cfg.data.num_workers = 0
    cfg.model.feature_size = 24
    cfg.model.dropout_path_rate = 0.0
    cfg.train.batch_size = 4
    cfg.train.amp = False
    cfg.train.edl_anneal_epochs = 2
    cfg.train.stages = (
        StageConfig("stage1_cls", 1, ("cls",)),
        StageConfig("stage2_pcad", 1, ("cls", "pcad"), freeze_backbone=True),
        StageConfig("stage3_uepd", 1, ("cls", "pcad", "uepd"), freeze_backbone=True),
        StageConfig("stage4_seg", 1, ("cls", "pcad", "uepd", "seg")),
        StageConfig("stage5_joint", 1, ("cls", "pcad", "uepd", "seg", "afdg")),
        StageConfig("stage6_afdg", 1, ("cls", "afdg"), freeze_backbone=True),
    )
    return cfg
