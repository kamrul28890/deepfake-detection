# Deepfake Detection Project Report

## Abstract

This project presents a reproducible deepfake detection workflow built around a public FaceForensics-style benchmark. The pipeline is intentionally layered: first it converts a public dataset into a sampled frame-level corpus, then it establishes a classical computer-vision baseline, then it fine-tunes a compact CNN, and finally it includes a small GAN module that can be used for augmentation experiments. The goal is not to produce a single black-box detector, but to create a paper-style research artifact that explains the problem, the data, the modeling choices, and the practical limitations in a way that is suitable for an MS-level project.

The central claim is simple: manipulated face imagery can often be identified through a combination of texture cues, local boundary artifacts, and transfer-learned spatial features, but the quality of the conclusion depends heavily on how the dataset is sampled and how the evaluation is interpreted. For example, a visually plausible fake frame can still carry high-frequency inconsistencies around the mouth or jawline that a hand-crafted texture descriptor may detect even when a human reviewer needs a second look.

## 1. Introduction

Deepfake detection has become a practical computer-vision task because modern face synthesis methods can generate outputs that look convincing to a casual observer. The task matters for media verification, platform integrity, and safety workflows. In this project, the problem is framed as binary classification at the frame level: given a face crop or frame, decide whether it is real or manipulated.

A frame-level formulation is useful because it reduces the computational burden and keeps the experimental setup transparent. It is also easy to explain. For example, if a short video contains 120 frames and only a subset of them are manipulated, a frame classifier can highlight suspicious segments before any temporal aggregation step is added later.

The repository is structured to support three goals at once:

- reproduce a small but valid deepfake benchmark from a public source
- compare classical and deep learning approaches on the same sampled subset
- provide a report artifact that explains the methodology and results in paper format

## 2. Problem Statement

The task is to infer whether an input face image is authentic or synthetic. The difficulty comes from the fact that many deepfakes preserve the broad facial geometry while altering local texture, shading, and blending patterns. That means a detector cannot rely on only one cue.

The project studies this question through three angles:

- Can hand-crafted descriptors such as HOG and color histograms separate real and fake frames?
- Does transfer learning with a modern CNN improve performance on the same sample?
- Can a GAN-based augmentation module support the training process by enriching the real-image distribution?

Example: if a frame is visually balanced but the hairline shows mild boundary smoothing, a shallow feature extractor may miss it while a CNN may still respond to the subtle local inconsistency.

## 3. Dataset

The default dataset source is a public FaceForensics-style dataset from Hugging Face. The code is written to sample a manageable subset and export a frame-level manifest so that the same items can be reused for experiments and report generation.

The repository emphasizes sample discipline because the full benchmark is large. Sampling matters for two reasons:

- it keeps the workflow runnable on a workstation
- it avoids the common mistake of reporting results from an uncontrolled subset

The prepared manifest records the source path, the exported frame path, and the label name. That makes the pipeline auditable. For example, if a frame is saved as `sample_00017.png`, the manifest allows the result table to connect that image back to its original dataset item.

Expected dataset structure:

- source: FaceForensicsC23-style Hugging Face dataset
- modality: video converted to representative frames
- labels: real and fake
- sampling: deterministic subset selection with a fixed random seed

## 4. Exploratory Data Analysis

The EDA step is not just a box-checking exercise. In this problem, the data distribution directly affects the interpretation of the results.

The following questions are most important:

- How imbalanced are the classes after sampling?
- Are the frames visually consistent in resolution and cropping?
- Do fake frames show stronger local smoothing or boundary blending than real frames?
- Is a particular manipulation mode disproportionately represented in the sample?

The codebase is designed to answer the first two questions directly from the manifest and the sampled images. The last two are answered visually and through downstream model behavior.

Example interpretation: if the fake class dominates the sample, a naive classifier can achieve high accuracy by predicting fake too often. In that situation, F1 and PR-AUC are more useful than accuracy because they reveal whether the minority class is being handled well.

Another practical example is frame quality. If the dataset contains both sharp and slightly blurred face crops, the model may learn resolution rather than manipulation. That is why resizing and normalization are explicit in the preprocessing pipeline.

## 5. Methodology

### 5.1 Frame sampling and preprocessing

The dataset loader extracts a representative frame from each sampled item and stores it locally. This transforms a video-oriented corpus into a reproducible image corpus.

The preprocessing steps are:

- load a public dataset split
- select a deterministic sample
- extract a representative frame
- save the frame as a PNG
- write a CSV manifest with labels and paths

Example: a sample can be re-created later simply by using the same dataset name, split, sample size, random seed, and frame index. This is essential for report reproducibility.

### 5.2 Classical baseline

The first baseline uses HOG features with per-channel color histograms. These are fed into logistic regression, a linear SVM, and a random forest.

Why this matters:

- HOG captures gradients and edge structure.
- Color histograms capture coarse tone and saturation patterns.
- Linear models provide a fast sanity check before deep training.

Example: if a fake frame has slightly smoother facial contours, the HOG descriptor may carry enough signal for a linear model to separate it from a real frame. If that happens, it suggests the dataset has measurable texture artifacts even before deep learning is introduced.

### 5.3 CNN fine-tuning

A ResNet18 backbone is fine-tuned on the same sampled frames. Transfer learning is appropriate because face images still share a lot with general photographic structure, so the pretrained filters can act as a strong starting point.

Technical settings:

- input size: 224 x 224
- optimizer: Adam
- learning rate: 1e-4
- loss: cross-entropy
- augmentation: horizontal flip and small rotation
- train/test split: stratified 80/20

Example: if the training set is small, a pretrained backbone is more stable than a model trained from scratch because the initial filters already know how to represent edges, corners, and mid-level contours.

### 5.4 GAN augmentation module

The GAN component is a compact DCGAN-style module that can learn from the real subset and synthesize additional face-like frames. It is included for two reasons:

- it gives the project a genuine GAN + CV component
- it lets the report discuss augmentation as an experimental extension rather than as a theoretical add-on

This module is best interpreted as an augmentation tool. Example: if the real subset does not include much lighting variation, synthetic samples can widen the appearance distribution and reduce overfitting. The GAN is not the detector; the detector remains the classifier.

## 6. Technical Parameters

The main implementation choices are summarized below:

- dataset source: bitmind/FaceForensicsC23 by default
- sample size: configurable, default 200 to 400 frames
- image representation: RGB frames with resizing and normalization
- classical features: HOG + 3-channel histograms
- deep model: ResNet18 fine-tuning
- GAN: DCGAN-style generator and discriminator trained on real frames
- random seed: fixed so the sampled subset is reproducible

Example: changing the sample size from 200 to 500 may improve model stability, but it also changes the class balance and the runtime. The report should always mention the sample size because it directly affects interpretability.

## 7. Results and Discussion

The repository writes result artifacts as JSON so they can be inserted into the report after a run. The important point is not only the raw score, but how the scores compare across methods.

The expected comparison pattern is:

- logistic regression gives a quick baseline
- linear SVM often performs similarly or slightly better when the HOG representation is strong
- random forest can be competitive but is often less stable on high-dimensional texture features
- ResNet18 should usually capture richer local structures and generalize better if the sample is large enough

Example interpretation: if the deep model improves recall more than precision, it may be capturing more fake frames but also introducing false positives. That is not automatically bad in a screening setting where missing a fake is more costly than checking an extra frame.

The GAN module should not be judged only by visual novelty. In this project it is evaluated as an augmentation mechanism. If the augmented training set yields a more stable validation curve or better recall on the minority class, then it serves its purpose.

## 8. Limitations

There are several important limitations.

- Frame-level detection ignores temporal inconsistencies across neighboring frames.
- A small sample can overstate deep-learning performance.
- A GAN trained on a limited subset can produce realistic-looking but not necessarily diverse samples.
- Results from one public source should not be assumed to transfer directly to other manipulation families.

Example: a model that separates one deepfake family well may still struggle on another because the visual artifacts differ. That is why the report should describe the dataset source explicitly rather than saying only “deepfake detection” in general terms.

## 9. Conclusion

This repository provides a complete deepfake detection workflow that is both technically credible and report-friendly. It includes public-data sampling, reproducible preprocessing, classical baselines, a deep transfer-learning model, and a GAN-based augmentation module. The design is intentionally simple enough to run on modest hardware while still being rich enough to support a detailed project report.

The main practical conclusion is that deepfake detection benefits from a layered approach. Simple texture descriptors are useful as sanity checks, transfer learning provides stronger representational power, and GAN-based augmentation can be used to improve data diversity or support further experiments.
