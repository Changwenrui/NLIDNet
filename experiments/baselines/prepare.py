import argparse
import csv
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from nlidnet.protocol import read_manifest, STAGES


def time_surfaces(events, start_frame, frames=20):
    t0 = events[0, 2] + start_frame * 2000
    selected = events[(events[:, 2] >= t0) & (events[:, 2] < t0 + 700 * 2000)]
    out = np.zeros((frames, 2, 600, 320), dtype=np.float32)
    gap = len(selected) // frames
    for i in range(frames):
        part = selected[i * gap : (i + 1) * gap]
        if not len(part):
            continue
        t = part[:, 2]
        dt = t.max() - t.min()
        norm = (t - t.min()) / dt if dt > 0 else np.zeros(len(t))
        values = (part[:, 3] * 2 - 1) * (2**norm - 1)
        valid = (
            (part[:, 0] >= 520)
            & (part[:, 0] < 840)
            & (part[:, 1] >= 0)
            & (part[:, 1] < 600)
        )
        coords = part[valid, :2].astype(int)
        values = values[valid]
        signed = np.zeros((600, 320), dtype=np.float32)

        if len(coords):
            flat = coords[:, 1] * 320 + coords[:, 0] - 520
            _, reverse = np.unique(flat[::-1], return_index=True)
            last = len(flat) - 1 - reverse
            signed.reshape(-1)[flat[last]] = values[last]
        image = np.stack([signed.clip(min=0), (-signed).clip(min=0)])
        scale = image.max() - image.min()
        out[i] = (image - image.min()) / scale if scale > 0 else image
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--event-root", required=True)
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = read_manifest(args.manifest)
    prepared_rows = []
    for r in rows:
        path = Path(args.event_root) / r["split"] / (r["prefix"] + ".safetensors")
        events = load_file(path)["events"]
        if not len(events):
            raise ValueError(f"Empty trial: {r['trial_id']}")
        dest = Path(args.output) / args.stage / r["split"] / (r["prefix"] + ".npy")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        np.save(dest, time_surfaces(events, STAGES[args.stage]))
        prepared_rows.append(
            {**r, "representation_path": f"{r['split']}/{r['prefix']}.npy"}
        )
        print(r["trial_id"], flush=True)
    fields = list(dict.fromkeys(key for row in prepared_rows for key in row))
    with (Path(args.output) / args.stage / "trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prepared_rows)


if __name__ == "__main__":
    main()
