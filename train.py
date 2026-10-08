import argparse

from src.config import Config, smoke_config
from src.trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description="GUIDE: occult GBC NC-CT diagnosis framework")
    ap.add_argument("--smoke", action="store_true", help="fast self-contained synthetic run")
    ap.add_argument("--data_root", default=None, help="root dir holding nc/pv/organ/lesion NIfTI volumes")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--case_manifest", default=None)
    ap.add_argument("--lesion_manifest", default=None)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else Config()
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.case_manifest:
        cfg.data.case_manifest = args.case_manifest
    if args.lesion_manifest:
        cfg.data.lesion_manifest = args.lesion_manifest

    Trainer(cfg, device=args.device).fit()


if __name__ == "__main__":
    main()
