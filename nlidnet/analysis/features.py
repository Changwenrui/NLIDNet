import csv
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass
class FeatureSet:
    features: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray


def load_features(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as obj:
            features = np.asarray(obj["features"], dtype=np.float32)
            labels = np.asarray(
                obj["raw_labels"] if "raw_labels" in obj else obj["labels"], dtype=int
            )
            ids = (
                np.asarray(obj["trial_ids"])
                if "trial_ids" in obj
                else np.asarray([f"{path.stem}/{i}" for i in range(len(labels))])
            )
    elif path.suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)

        def array(value):
            return (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )

        features = array(obj["features"]).astype(np.float32)
        labels = array(
            obj["raw_labels"] if "raw_labels" in obj else obj["labels"]
        ).astype(int)
        ids = (
            array(obj["trial_ids"])
            if "trial_ids" in obj
            else np.asarray(
                [f"{path.stem}/{v}" for v in obj.get("prefixes", range(len(labels)))]
            )
        )
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            names = reader.fieldnames or []
            columns = [
                n for n in names if re.fullmatch(r"(?:feature|feat|f)[_-]?\d+", n)
            ]
            columns = columns or names[3:]
            rows = list(reader)
        features = np.asarray(
            [[float(r[c]) for c in columns] for r in rows], dtype=np.float32
        )
        label_key = next(
            (n for n in ["raw_label", "label", "class_label"] if n in names), names[1]
        )
        id_key = next((n for n in ["trial_id", "prefix"] if n in names), names[0])
        labels = np.asarray([int(r[label_key]) for r in rows])
        ids = np.asarray([r[id_key] for r in rows])
    else:
        raise ValueError("Feature files must be NPZ, PT or CSV")
    if (
        features.ndim != 2
        or not len(features)
        or len(features) != len(labels)
        or len(labels) != len(ids)
    ):
        raise ValueError(f"Invalid feature arrays: {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features: {path}")
    return FeatureSet(features, labels.reshape(-1), ids.astype(str))


def combine_features(paths):
    items = [load_features(p) for p in paths]
    return FeatureSet(
        np.concatenate([x.features for x in items]),
        np.concatenate([x.labels for x in items]),
        np.concatenate([x.trial_ids for x in items]),
    )
