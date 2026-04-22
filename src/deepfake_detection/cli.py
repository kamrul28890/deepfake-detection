from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .dataset import DatasetConfig, prepare_frame_dataset
from .features import build_feature_matrix, load_manifest
from .gan import train_dcgan
from .models import train_classical_baselines, train_resnet18
from .reporting import write_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _load_manifest() -> pd.DataFrame:
    return load_manifest(DATA_DIR / "manifest.csv")


def command_prepare_data(args: argparse.Namespace) -> None:
    config = DatasetConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        frame_index=args.frame_index,
    )
    manifest_path = prepare_frame_dataset(DATA_DIR, config)
    print(manifest_path)


def command_run_baselines(_: argparse.Namespace) -> None:
    manifest = _load_manifest()
    feature_matrix = build_feature_matrix(manifest["frame_path"].tolist())
    labels = manifest["label"].astype(int).to_numpy()
    results = train_classical_baselines(feature_matrix, labels)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        {"model": result.name, **result.metrics}
        for result in results
    ]
    (ARTIFACTS_DIR / "baseline_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(ARTIFACTS_DIR / "baseline_metrics.json")


def command_train_deep(args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    result = train_resnet18(
        manifest,
        output_dir=ARTIFACTS_DIR / "deep",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    (ARTIFACTS_DIR / "deep_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(ARTIFACTS_DIR / "deep_metrics.json")


def command_train_gan(args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    real_images = manifest.loc[manifest["label"].astype(int) == 0, "frame_path"].tolist()
    result = train_dcgan(
        real_images,
        output_dir=ARTIFACTS_DIR / "gan",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    payload = {
        "generator_path": str(result.generator_path),
        "sample_grid_path": str(result.sample_grid_path),
    }
    (ARTIFACTS_DIR / "gan_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(ARTIFACTS_DIR / "gan_result.json")


def command_write_report(_: argparse.Namespace) -> None:
    context = {
        "dataset_summary": {
            "source": "bitmind/FaceForensicsC23",
            "sample_size": len(_load_manifest()),
            "real_count": int((_load_manifest()["label"].astype(int) == 0).sum()),
            "fake_count": int((_load_manifest()["label"].astype(int) == 1).sum()),
        },
        "baseline_results": json.loads((ARTIFACTS_DIR / "baseline_metrics.json").read_text(encoding="utf-8")) if (ARTIFACTS_DIR / "baseline_metrics.json").exists() else [],
        "deep_metrics": json.loads((ARTIFACTS_DIR / "deep_metrics.json").read_text(encoding="utf-8")) if (ARTIFACTS_DIR / "deep_metrics.json").exists() else {},
        "gan_result": json.loads((ARTIFACTS_DIR / "gan_result.json").read_text(encoding="utf-8")) if (ARTIFACTS_DIR / "gan_result.json").exists() else {},
    }
    report_path = write_report(REPORTS_DIR / "final_project_report.md", context)
    print(report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deepfake detection project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Sample frames from the public dataset")
    prepare.add_argument("--dataset-name", default="bitmind/FaceForensicsC23")
    prepare.add_argument("--split", default="train")
    prepare.add_argument("--sample-size", type=int, default=200)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--frame-index", type=int, default=0)
    prepare.set_defaults(func=command_prepare_data)

    baselines = subparsers.add_parser("run-baselines", help="Train classical CV baselines")
    baselines.set_defaults(func=command_run_baselines)

    deep = subparsers.add_parser("train-deep", help="Fine-tune the ResNet18 classifier")
    deep.add_argument("--epochs", type=int, default=1)
    deep.add_argument("--batch-size", type=int, default=16)
    deep.add_argument("--learning-rate", type=float, default=1e-4)
    deep.set_defaults(func=command_train_deep)

    gan = subparsers.add_parser("train-gan", help="Train the DCGAN augmentation module")
    gan.add_argument("--epochs", type=int, default=1)
    gan.add_argument("--batch-size", type=int, default=32)
    gan.set_defaults(func=command_train_gan)

    report = subparsers.add_parser("write-report", help="Generate the markdown report")
    report.set_defaults(func=command_write_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
