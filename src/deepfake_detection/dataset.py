from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import Video, load_dataset
from PIL import Image


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str = "bitmind/FaceForensicsC23"
    split: str = "train"
    sample_size: int = 400
    seed: int = 42
    frame_index: int = 0


IMAGE_KEYS = ("image", "img", "frame", "frames")
VIDEO_KEYS = ("video", "path", "file", "filepath", "media")
LABEL_KEYS = ("label", "labels", "target", "class", "is_fake")


def _pick_key(example: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in example:
            return key
    raise KeyError(f"Could not infer one of {candidates} from example keys: {list(example.keys())}")


def _label_name(features: Any, label_key: str, value: Any) -> str:
    try:
        label_feature = features[label_key]
        if hasattr(label_feature, "names") and label_feature.names:
            index = int(value)
            return str(label_feature.names[index])
    except Exception:
        pass
    if isinstance(value, bool):
        return "fake" if value else "real"
    return str(value)


def _decode_image_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


def _decode_video_frame(value: Any, frame_index: int) -> Image.Image:
    if isinstance(value, dict) and value.get("path"):
        video_path = value["path"]
    elif isinstance(value, dict) and value.get("bytes") is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(value["bytes"])
            video_path = temp_file.name
    elif isinstance(value, str):
        video_path = value
    else:
        raise TypeError(f"Unsupported video payload type: {type(value)!r}")

    if isinstance(video_path, str) and video_path.startswith("zip://"):
        member_path, archive_path = video_path.removeprefix("zip://").split("::", maxsplit=1)
        with zipfile.ZipFile(archive_path) as archive:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(archive.read(member_path))
                video_path = temp_file.name

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _source_path(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("path", ""))
    if isinstance(value, str):
        return value
    return ""


def _infer_binary_label_from_path(source_path: str) -> tuple[int, str]:
    normalized = source_path.replace("\\", "/").lower()
    parts = normalized.split("/")
    if "real" in parts:
        return 0, "real"
    if "fake" in parts:
        return 1, "fake"
    raise KeyError(f"Could not infer real/fake label from source path: {source_path}")


def load_dataset_subset(config: DatasetConfig):
    dataset = load_dataset(config.dataset_name, split=config.split)
    for key in VIDEO_KEYS:
        if key in dataset.column_names:
            dataset = dataset.cast_column(key, Video(decode=False))
            break
    sample_size = min(config.sample_size, len(dataset))
    indices = np.random.default_rng(config.seed).choice(len(dataset), size=sample_size, replace=False)
    return dataset.select(sorted(int(index) for index in indices))


def prepare_frame_dataset(
    output_dir: str | Path,
    config: DatasetConfig | None = None,
) -> Path:
    config = config or DatasetConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_subset(config)
    column_names = set(dataset.column_names)
    image_key = next((key for key in IMAGE_KEYS if key in column_names), None)
    video_key = next((key for key in VIDEO_KEYS if key in column_names), None)
    label_key = next((key for key in LABEL_KEYS if key in column_names), None)

    if image_key is None and video_key is None:
        raise KeyError(f"Could not infer an image/video key from dataset columns: {dataset.column_names}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "source_path", "frame_path", "label", "label_name"],
        )
        writer.writeheader()

        for index, row in enumerate(dataset):
            source_value = row.get(image_key) if image_key else row.get(video_key)
            if image_key:
                image = _decode_image_value(source_value)
            else:
                image = _decode_video_frame(source_value, config.frame_index)

            if label_key is None:
                label_value, label_name = _infer_binary_label_from_path(_source_path(source_value))
            else:
                label_value = row[label_key]
                label_name = _label_name(dataset.features, label_key, label_value)
            class_dir = output_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)
            frame_path = class_dir / f"sample_{index:05d}.png"
            image.save(frame_path)

            source_path = _source_path(source_value)

            writer.writerow(
                {
                    "sample_id": index,
                    "source_path": source_path,
                    "frame_path": str(frame_path),
                    "label": label_value,
                    "label_name": label_name,
                }
            )

    return manifest_path
