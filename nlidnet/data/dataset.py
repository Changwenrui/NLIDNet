from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file
from nlidnet.protocol import read_manifest, BASE_LABELS
from nlidnet.data.physics import get_normalized_phys_target


class LiquidEventDataset(Dataset):
    def __init__(self, cfg, split, labels=BASE_LABELS, offset=None, records=None):
        self.cfg, self.split, self.offset = cfg, split, offset
        self.allowed_labels = tuple(labels)
        self.label_map = {v: i for i, v in enumerate(self.allowed_labels)}
        rows = read_manifest(cfg["manifest"]) if records is None else records
        self.records = [
            r for r in rows if r["split"] == split and r["label"] in self.label_map
        ]
        if not self.records:
            raise ValueError(f"No {split} trials for labels {labels}")
        self.label_to_indices = {label: [] for label in self.allowed_labels}
        for i, r in enumerate(self.records):
            self.label_to_indices[r["label"]].append(i)
            for root, ext in [
                (cfg["event_root"], "safetensors"),
                (cfg["roi_root"], "pt"),
            ]:
                p = Path(root) / split / (r["prefix"] + "." + ext)
                if not p.is_file():
                    raise FileNotFoundError(p)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        cfg = self.cfg
        offset = (
            self.offset
            if self.offset is not None
            else random.randrange(cfg["train_offset_count"])
        )
        start = cfg["start_frame"] + offset
        signal = torch.load(
            Path(cfg["roi_root"]) / self.split / (r["prefix"] + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        if (
            signal.ndim != 2
            or signal.shape[0] != 6
            or signal.shape[1] < start + cfg["window_bins"]
        ):
            raise ValueError(
                f"Invalid ROI signal for trial {r['trial_id']}: {tuple(signal.shape)}"
            )
        events = load_file(
            str(Path(cfg["event_root"]) / self.split / (r["prefix"] + ".safetensors")),
            device="cpu",
        )["events"]
        if events.ndim != 2 or events.shape[1] != 4 or not len(events):
            raise ValueError(f"Expected nonempty [N,4] events: {r['trial_id']}")

        ts = events[:, 2].double()
        t_start = ts[0].item() + start * cfg["delta_t_us"]
        keep = (ts >= t_start) & (ts < t_start + cfg["window_bins"] * cfg["delta_t_us"])
        selected = events[keep][: cfg["max_events"]]
        cropped = selected.float().clone()
        if len(cropped):
            cropped[:, 2] = (selected[:, 2].double() - t_start).float()

        phys = (
            get_normalized_phys_target(r["label"])
            if r["label"] in BASE_LABELS
            else torch.zeros(3)
        )
        return {
            "signal": signal[:, start : start + cfg["window_bins"]].float(),
            "events": cropped,
            "label": self.label_map[r["label"]],
            "raw_label": r["label"],
            "physics": phys,
            "trial_id": r["trial_id"],
        }


def collate_trials(items):
    n = max(1, max(len(x["events"]) for x in items))
    events = torch.zeros(len(items), n, 4)
    mask = torch.zeros(len(items), n)
    for i, x in enumerate(items):
        m = len(x["events"])
        events[i, :m] = x["events"]
        mask[i, :m] = 1
    return {
        "inputs": (
            torch.stack([x["signal"] for x in items]),
            events,
            mask,
            torch.zeros(len(items)),
        ),
        "labels": torch.tensor([x["label"] for x in items]),
        "raw_labels": torch.tensor([x["raw_label"] for x in items]),
        "physics": torch.stack([x["physics"] for x in items]),
        "trial_ids": [x["trial_id"] for x in items],
    }
