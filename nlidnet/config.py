import json
from pathlib import Path
import yaml
from nlidnet.protocol import STAGES


def load_config(path, stage=None):
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if stage is not None:
        cfg["stage"] = stage
    cfg["start_frame"] = STAGES[cfg["stage"]]
    if cfg["T"] * cfg["k"] != cfg["window_bins"]:
        raise ValueError("T * k must equal window_bins")
    if cfg["classes_per_batch"] * cfg["samples_per_class"] != cfg["batch_size"]:
        raise ValueError("Balanced sampler dimensions must match batch_size")
    for name in ["epochs", "batch_size", "T", "k", "delta_t_us", "max_events"]:
        if cfg[name] < 1:
            raise ValueError(f"{name} must be positive")
    return cfg


def save_config(cfg, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
