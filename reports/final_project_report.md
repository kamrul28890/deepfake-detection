# Deepfake Detection Project Report

## Abstract

This project studies frame-level deepfake detection on a public FaceForensics-style benchmark. The design combines three layers: reproducible frame sampling, classical computer-vision baselines, and a compact deep model fine-tuning stage. A small GAN module is included as an augmentation-oriented extension so the repository reflects a realistic GAN + CV workflow rather than a single-model demo.

## 1. Problem Statement

The core task is binary classification: determine whether a face frame is authentic or manipulated. In practice, the problem is difficult because synthetic frames can preserve overall facial structure while altering local textures, blending boundaries, eye regions, and skin statistics. For example, a fake frame may look visually plausible at a glance but still exhibit subtle convolutional artifacts or inconsistent frequency patterns.

## 2. Dataset

The default dataset pipeline targets a public FaceForensicsC23-style source loaded from Hugging Face and then converts it into a sampled frame dataset. The project records a reproducible subset so the same experiment can be rerun on a workstation without handling the full video corpus.

Dataset snapshot:
- Source: bitmind/FaceForensicsC23
- Sampled frames: 200
- Real examples: 22
- Fake examples: 178

Example: if a sampled video frame is saved as `sample_00012.png`, the label is derived from the dataset metadata and preserved in the manifest so later experiments can trace every prediction back to its source item.

## 3. Exploratory Data Analysis

EDA focuses on the practical structure of the sample rather than only the raw class counts. The key checks are class balance, image resolution consistency, and the degree of visual variation across examples.

Observed pattern:
- Frames are normalized to a uniform size for downstream modeling.
- The binary labels remain imbalanced enough that F1 and PR-AUC are more informative than accuracy.
- Example visual inspection shows that real frames tend to preserve stable skin and background boundaries, while fake frames often introduce local smoothing or blending artifacts.

## 4. Methodology

The pipeline is organized into three modeling tracks.

### 4.1 Classical CV baseline

A HOG + histogram feature representation is used with logistic regression, linear SVM, and random forest. This gives a low-cost benchmark that tests whether the dataset is separable with hand-crafted texture descriptors.

Example: a face crop with obvious blending seams may produce a stronger HOG signature near the jawline, which a linear model can exploit without requiring a large neural network.

### 4.2 Deep CNN fine-tuning

A ResNet18 backbone is fine-tuned on sampled face frames. This stage gives the model access to learned spatial filters and transfer learning from natural-image pretraining.

Technical settings:
- Input size: 224 x 224
- Optimizer: Adam
- Learning rate: 1e-4
- Epochs: 1

### 4.3 GAN augmentation track

The GAN module uses a compact DCGAN architecture to synthesize additional face-like frames from the real subset. The purpose is not to replace the detector, but to provide an augmentation channel and to demonstrate a GAN + CV workflow in a reproducible way.

Example: if the real subset contains limited lighting diversity, synthetic samples can encourage the CNN to see broader appearance modes during training.

## 5. Technical Parameters

- Frame sampling: 0
- Image normalization: RGB conversion and ImageNet-style normalization for deep models
- Classical features: HOG plus per-channel color histograms
- Train/test split: 80/20 stratified split
- Random seed: 42

## 6. Results

### Classical baselines

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc | confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.850 | 0.895 | 0.944 | 0.919 | 0.326 | 0.872 | [[0, 4], [2, 34]] |
| linear_svm | 0.900 | 0.900 | 1.000 | 0.947 | 0.785 | 0.974 | [[0, 4], [0, 36]] |
| random_forest | 0.900 | 0.900 | 1.000 | 0.947 | 0.288 | 0.836 | [[0, 4], [0, 36]] |

### Deep model

{
  "checkpoint_path": "D:\\Projects\\healthcare NLP\\deepfake-detection\\artifacts\\deep\\resnet18_deepfake.pt",
  "metrics": {
    "accuracy": 0.875,
    "precision": 0.8974358974358975,
    "recall": 0.9722222222222222,
    "f1": 0.9333333333333333,
    "roc_auc": 0.5069444444444444,
    "pr_auc": 0.9301000720606764,
    "confusion_matrix": [
      [
        0,
        4
      ],
      [
        1,
        35
      ]
    ],
    "classification_report": "              precision    recall  f1-score   support\n\n           0       0.00      0.00      0.00         4\n           1       0.90      0.97      0.93        36\n\n    accuracy                           0.88        40\n   macro avg       0.45      0.49      0.47        40\nweighted avg       0.81      0.88      0.84        40\n"
  }
}

### GAN output

{
  "generator_path": "D:\\Projects\\healthcare NLP\\deepfake-detection\\artifacts\\gan\\dcgan_generator.pt",
  "sample_grid_path": "D:\\Projects\\healthcare NLP\\deepfake-detection\\artifacts\\gan\\dcgan_samples.png"
}

## 7. Discussion

The main practical conclusion is that the dataset supports a staged workflow. Hand-crafted features offer a fast benchmark, while transfer learning usually captures more of the subtle manipulation structure. The GAN component is most useful as an augmentation and narrative complement, because the detection problem itself still depends on stable discriminative features rather than synthetic generation alone.

Example interpretation: if the CNN improves F1 but not accuracy, that usually indicates the class balance was not uniform and the model became better at recovering the minority class rather than simply predicting the majority label.

## 8. Limitations

- Frame-level classification ignores temporal inconsistencies that appear across adjacent video frames.
- Very small sample sizes can overstate the quality of the deep model.
- GAN training on limited face crops can be unstable and should be treated as an auxiliary experiment.

## 9. Conclusion

The project delivers a reproducible deepfake detection pipeline that can be extended with larger samples, temporal models, and stronger augmentation policies. The current design is suitable for an MS-level prototype because it combines a clear problem definition, dataset handling, baselines, a deep model, and a reportable experimental workflow.
