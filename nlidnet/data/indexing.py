import csv
import re
from collections import Counter
from pathlib import Path
from nlidnet.protocol import LIQUIDS, validate_manifest_rows


def natural_key(path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(path))
    ]


def write_manifest(rows, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not rows:
        raise ValueError("No trials found")
    validate_manifest_rows(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def index_raw(raw_root, output, split_file=None, test_per_class=None, folder_map=None):
    root = Path(raw_root)
    if (split_file is None) == (test_per_class is None):
        raise ValueError("Specify either --split-file or --test-per-class")
    if test_per_class is not None and test_per_class < 1:
        raise ValueError("test-per-class must be positive")
    explicit = None
    if split_file is not None:
        with Path(split_file).open(encoding="utf-8-sig", newline="") as stream:
            split_rows = list(csv.DictReader(stream))
        explicit = {}
        for row in split_rows:
            name = Path(row["raw_path"]).as_posix()
            if name in explicit or row["split"] not in {"train", "test"}:
                raise ValueError(f"Duplicate or invalid split assignment: {row}")
            explicit[name] = row
    folders = {liquid: liquid for liquid in LIQUIDS}
    if folder_map:
        folders.update(folder_map)
    rows = []
    used = set()
    for label, liquid in enumerate(LIQUIDS):
        folder = root / folders[liquid]
        files = sorted(folder.glob("*.raw"), key=lambda p: natural_key(p.name))
        if not files:
            continue
        for rank, path in enumerate(files):
            raw_path = path.relative_to(root).as_posix()
            if explicit is not None:
                if raw_path not in explicit:
                    raise ValueError(f"Missing split assignment: {raw_path}")
                source = explicit[raw_path]
                split = source["split"]
                trial_id = source.get("trial_id") or f"{liquid}_{path.stem}"
                prefix = source.get("prefix") or trial_id
                used.add(raw_path)
            else:
                split = "test" if rank < test_per_class else "train"
                trial_id = prefix = f"{liquid}_{path.stem}"
            rows.append(
                dict(
                    trial_id=trial_id,
                    split=split,
                    label=label,
                    prefix=prefix,
                    raw_path=raw_path,
                    liquid=liquid,
                )
            )
    if explicit is not None and used != set(explicit):
        raise ValueError(
            f"Split file references missing recordings: {sorted(set(explicit) - used)[:10]}"
        )
    return write_manifest(rows, output)


def index_processed(event_root, output, roi_root=None):
    root = Path(event_root)
    roi_root = Path(roi_root or event_root)
    rows = []
    for split in ("train", "test"):
        for path in sorted(
            (root / split).glob("*.safetensors"), key=lambda p: natural_key(p.name)
        ):
            prefix = path.stem
            label_path = root / split / f"{prefix}.txt"
            if not label_path.is_file():
                label_path = roi_root / split / f"{prefix}.txt"
            label = int(label_path.read_text(encoding="utf-8-sig").strip())
            if not 0 <= label < len(LIQUIDS):
                raise ValueError(f"Unknown liquid label {label}: {label_path}")
            if not (roi_root / split / f"{prefix}.pt").is_file():
                raise FileNotFoundError(roi_root / split / f"{prefix}.pt")
            rows.append(
                dict(
                    trial_id=f"{split}_{prefix}",
                    split=split,
                    label=label,
                    prefix=prefix,
                    liquid=LIQUIDS[label],
                )
            )
    return write_manifest(rows, output)


def manifest_summary(rows):
    return {
        f"{split}/{LIQUIDS[label]}": count
        for (split, label), count in sorted(
            Counter((r["split"], r["label"]) for r in rows).items()
        )
    }
