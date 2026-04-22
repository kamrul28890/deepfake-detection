from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .evaluation import evaluate_binary_predictions


@dataclass
class ClassicalResult:
    name: str
    metrics: dict[str, Any]


class FrameClassificationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 224, augment: bool = False):
        if "label" not in manifest.columns:
            raise KeyError("manifest must contain a label column")
        self.manifest = manifest.reset_index(drop=True)
        transform_list = [transforms.Resize((image_size, image_size))]
        if augment:
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5),
                ]
            )
        transform_list.extend([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image = Image.open(row.frame_path).convert("RGB")
        image_tensor = self.transform(image)
        label = int(row.label)
        return image_tensor, label


def prepare_binary_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    working = manifest.copy()
    if working["label"].dtype == object:
        mapping = {name: index for index, name in enumerate(sorted(working["label_name"].unique()))}
        working["label"] = working["label_name"].map(mapping).astype(int)
    else:
        working["label"] = working["label"].astype(int)
    if working["label"].nunique() > 2:
        first_label = sorted(working["label"].unique())[0]
        working = working.assign(label=(working["label"] != first_label).astype(int))
    return working


def train_classical_baselines(feature_matrix: np.ndarray, labels: np.ndarray, test_size: float = 0.2, seed: int = 42):
    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    models_map = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": SVC(kernel="linear", probability=True, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample"),
    }

    results: list[ClassicalResult] = []
    for name, estimator in models_map.items():
        estimator.fit(x_train, y_train)
        if hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(x_test)[:, 1]
        else:
            decision = estimator.decision_function(x_test)
            scores = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
        metrics = evaluate_binary_predictions(y_test, scores)
        results.append(ClassicalResult(name=name, metrics=metrics))
    return results


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    network = models.resnet18(weights=weights)
    network.fc = nn.Linear(network.fc.in_features, num_classes)
    return network


def train_resnet18(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working = prepare_binary_labels(manifest)
    train_frame, test_frame = train_test_split(working, test_size=test_size, random_state=seed, stratify=working["label"])

    train_dataset = FrameClassificationDataset(train_frame, augment=True)
    test_dataset = FrameClassificationDataset(test_frame, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = build_resnet18(num_classes=2, pretrained=True).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        network.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    network.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = network(images)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_score.extend(scores.tolist())
            y_true.extend(labels.numpy().tolist())

    metrics = evaluate_binary_predictions(np.asarray(y_true), np.asarray(y_score))
    checkpoint_path = output_dir / "resnet18_deepfake.pt"
    torch.save(network.state_dict(), checkpoint_path)
    return {"checkpoint_path": str(checkpoint_path), "metrics": metrics}
