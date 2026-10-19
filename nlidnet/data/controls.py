import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from nlidnet.protocol import BASE_LABELS, CONCENTRATIONS, read_manifest


class ROISignalDataset(Dataset):
    def __init__(
        self,
        cfg,
        split,
        task="classification",
        offset=None,
        channels=None,
        normalize=False,
    ):
        self.cfg, self.split, self.task, self.offset = cfg, split, task, offset
        self.channels, self.normalize = channels, normalize
        labels = tuple(CONCENTRATIONS) if task == "regression" else BASE_LABELS
        self.label_map = {label: index for index, label in enumerate(labels)}
        self.rows = [
            r
            for r in read_manifest(cfg["manifest"])
            if r["split"] == split and r["label"] in labels
        ]
        if not self.rows:
            raise ValueError(f"No {split} trials for {task}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(self.cfg["roi_root"]) / self.split / (row["prefix"] + ".pt")
        signal = torch.load(path, map_location="cpu", weights_only=True).float()
        offset = (
            self.offset
            if self.offset is not None
            else random.randrange(self.cfg["train_offset_count"])
        )
        start = self.cfg["start_frame"] + offset
        if (
            signal.ndim != 2
            or signal.shape[0] != 6
            or signal.shape[1] < start + self.cfg["window_bins"]
        ):
            raise ValueError(f"Invalid six-ROI signal: {path}")
        signal = signal[:, start : start + self.cfg["window_bins"]]
        if self.channels is not None:
            signal = signal[self.channels]
        if self.normalize:
            signal = (signal - signal.mean(dim=1, keepdim=True)) / signal.std(
                dim=1, keepdim=True, correction=0
            ).clamp_min(1e-6)
        target = (
            CONCENTRATIONS[row["label"]] / 100
            if self.task == "regression"
            else self.label_map[row["label"]]
        )
        return signal, target, row["trial_id"]


class RGBFrameDataset(Dataset):
    def __init__(
        self,
        manifest,
        video_root,
        split,
        mode="first_frame",
        seed=42,
        pour_start=5.9,
        pour_end=10.1,
        grayscale=False,
    ):
        if (
            mode not in {"first_frame", "last_frame", "random_pour"}
            or pour_end <= pour_start
        ):
            raise ValueError("Invalid RGB frame selection")
        self.rows = [
            r
            for r in read_manifest(manifest)
            if r["split"] == split and r["label"] in BASE_LABELS
        ]
        self.label_map = {label: index for index, label in enumerate(BASE_LABELS)}
        self.root, self.split, self.mode, self.seed = (
            Path(video_root),
            split,
            mode,
            seed,
        )
        self.start, self.end, self.grayscale = pour_start, pour_end, grayscale
        for row in self.rows:
            if (
                not row.get("video_path")
                or not (self.root / row["video_path"]).is_file()
            ):
                raise FileNotFoundError(f"Missing video_path for {row['trial_id']}")
        if not self.rows:
            raise ValueError(f"No {split} RGB trials")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2

        row = self.rows[index]
        cap = cv2.VideoCapture(str(self.root / row["video_path"]))
        try:
            fps, total = cap.get(cv2.CAP_PROP_FPS), int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            if fps <= 0 or total < 1:
                raise ValueError(f"Invalid RGB recording: {row['trial_id']}")
            if self.mode == "first_frame":
                frame_index = 0
            elif self.mode == "last_frame":
                frame_index = total - 1
            else:
                start = min(int(self.start * fps), total - 1)
                end = min(max(start + 1, int(self.end * fps)), total)
                rng = (
                    random
                    if self.split == "train"
                    else random.Random(self.seed + index * 10007)
                )
                frame_index = rng.randrange(start, end)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Cannot decode frame {frame_index}: {row['trial_id']}"
                )
            if self.grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            image = (
                torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()
                / 255
            )
            image = (
                image - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            return image, self.label_map[row["label"]], row["trial_id"]
        finally:
            cap.release()
