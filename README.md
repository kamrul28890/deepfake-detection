# Deepfake Detection

This project benchmarks deepfake detection on a public FaceForensics-style dataset with three complementary pieces:

- a reproducible frame-sampling data pipeline
- classical CV baselines using HOG features
- a compact CNN fine-tuner and a small GAN-based augmentation module

## Dataset

The default loader targets a public FaceForensicsC23-style dataset from Hugging Face. The code samples a smaller, reproducible frame-level subset so the repository can be run on a modest workstation.

## Main commands

```bash
python -m deepfake_detection.cli prepare-data
python -m deepfake_detection.cli run-baselines
python -m deepfake_detection.cli train-deep
python -m deepfake_detection.cli train-gan
python -m deepfake_detection.cli write-report
```

## Outputs

- `data/sample/manifest.csv`
- `artifacts/metrics.json`
- `artifacts/figures/`
- `reports/final_project_report.md`

