from pathlib import Path
from nlidnet.protocol import read_manifest


def validate_data(cfg):
    rows = read_manifest(cfg["manifest"])
    failures = []
    for r in rows:
        for key, ext in [("event_root", "safetensors"), ("roi_root", "pt")]:
            p = Path(cfg[key]) / r["split"] / (r["prefix"] + "." + ext)
            if not p.is_file():
                failures.append(str(p))
    if failures:
        raise FileNotFoundError("Missing paired tensors:\n" + "\n".join(failures[:20]))
    print(
        {
            "trials": len(rows),
            "train": sum(r["split"] == "train" for r in rows),
            "test": sum(r["split"] == "test" for r in rows),
        }
    )
