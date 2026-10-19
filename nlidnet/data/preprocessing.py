from pathlib import Path
import numpy as np
from nlidnet.protocol import ROI_BOXES, read_manifest
from nlidnet.data.raw import read_events


def roi_counts(events, delta_t_us=2000, bins=9000):
    if delta_t_us <= 0 or bins <= 0:
        raise ValueError("delta_t_us and bins must be positive")
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Expected [N,4] x,y,t,p events")
    out = np.zeros((6, bins), dtype=np.float32)
    if not len(events):
        return out
    t = ((events[:, 2] - events[0, 2]) // delta_t_us).astype(np.int64)
    for i, (x0, y0, x1, y1) in enumerate(ROI_BOXES):
        keep = (
            (events[:, 0] >= x0)
            & (events[:, 0] < x1)
            & (events[:, 1] >= y0)
            & (events[:, 1] < y1)
            & (t >= 0)
            & (t < bins)
        )
        out[i] = np.bincount(t[keep], minlength=bins)
    return out


def read_raw(path):
    result = read_events(path)
    if not len(result):
        raise ValueError(f"Empty recording: {path}")
    return result


def preprocess(
    manifest, raw_root, output, delta_t_us=2000, bins=9000, labels=None, splits=None
):
    import torch
    from safetensors.torch import save_file

    records = [
        r
        for r in read_manifest(manifest)
        if (labels is None or r["label"] in labels)
        and (splits is None or r["split"] in splits)
    ]
    raw_paths = []
    for r in records:
        if not r.get("raw_path"):
            raise ValueError("Preprocessing requires a raw_path column")
        raw_paths.append((Path(raw_root) / r["raw_path"]).resolve())
    if len(raw_paths) != len(set(raw_paths)):
        raise ValueError(
            "The same RAW recording appears more than once in the manifest"
        )
    for r, raw in zip(records, raw_paths):
        dest = Path(output) / r["split"] / r["prefix"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        targets = [Path(str(dest) + ext) for ext in [".pt", ".safetensors", ".txt"]]
        if any(p.exists() for p in targets):
            raise FileExistsError(f"Refusing to overwrite prepared trial: {dest}")
        events = read_raw(raw)
        torch.save(torch.from_numpy(roi_counts(events, delta_t_us, bins)), targets[0])
        save_file({"events": torch.from_numpy(events.copy())}, str(targets[1]))
        targets[2].write_text(str(r["label"]), encoding="utf-8")
        print(r["trial_id"])
