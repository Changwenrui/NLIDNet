from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from nlidnet.protocol import read_manifest, BASE_LABELS


class BaselineDataset(Dataset):
    def __init__(
        self, manifest, root, split, model_name, start_seconds=5.9, duration=1.4
    ):
        rows = read_manifest(manifest)
        self.rows = [
            r for r in rows if r["split"] == split and r["label"] in BASE_LABELS
        ]
        self.root = Path(root)
        self.name = model_name
        self.start, self.duration = start_seconds, duration
        self.map = {label: i for i, label in enumerate(BASE_LABELS)}
        self.video = model_name in {
            "videomae_v2",
            "go_with_the_flow",
            "dont_look_twice",
        }
        field = "video_path" if self.video else "representation_path"
        for r in self.rows:
            if not r.get(field) or not (self.root / r[field]).is_file():
                raise FileNotFoundError(f"Missing {field} for {r['trial_id']}")
        if not self.rows:
            raise ValueError(f"No base-category {split} trials")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.video:
            import cv2

            cap = cv2.VideoCapture(str(self.root / r["video_path"]))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    raise ValueError(f"Invalid frame rate: {r['trial_id']}")
                indices = np.linspace(
                    int(self.start * fps),
                    int((self.start + self.duration) * fps) - 1,
                    16,
                ).astype(int)
                frames = []
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(
                            f"Incomplete video for {r['trial_id']} at frame {index}"
                        )
                    rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
                    frames.append(
                        torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
                    )
                x = torch.stack(frames, dim=1)
            finally:
                cap.release()
        else:
            array = np.load(self.root / r["representation_path"], allow_pickle=False)
            if array.ndim != 4 or array.shape[:2] != (20, 2):
                raise ValueError("Event baseline representation must be [20,2,H,W]")
            x = torch.from_numpy(array).float()
            if self.name in {"alexnet_lstm", "mobilenetv3_chem"}:
                x = F.interpolate(
                    x, size=(300, 160), mode="bilinear", align_corners=False
                )
            x = x.permute(1, 0, 2, 3)
        return x, self.map[r["label"]]
