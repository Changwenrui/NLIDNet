import json
import random
from pathlib import Path
import numpy as np
import torch
from nlidnet.models import NLIDNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg, checkpoint=None):
    kwargs = dict(T=cfg["T"], k=cfg["k"], delta_t=cfg["delta_t_us"])
    if checkpoint is None:
        for key in ["resnet_checkpoint", "flow_checkpoint"]:
            if not cfg.get(key) or not Path(cfg[key]).is_file():
                raise FileNotFoundError(f"{key}: {cfg.get(key)} (see README.md)")
        kwargs.update(
            resnet_checkpoint=cfg["resnet_checkpoint"],
            flow_checkpoint=cfg["flow_checkpoint"],
        )
    model = NLIDNet(**kwargs)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state.get("state_dict", state))
        clean = {}
        for key, value in state.items():
            for prefix in ["module.", "_orig_mod."]:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            clean[key] = value
        model.load_state_dict(clean, strict=True)
    return model


def move_inputs(batch, device):
    return tuple(t.to(device) for t in batch["inputs"])


def aggregate_trials(values, labels, trial_ids):
    grouped = {}
    for value, label, trial_id in zip(values, labels, trial_ids):
        if trial_id in grouped:
            total, n, previous = grouped[trial_id]
            if previous != label:
                raise ValueError(f"Inconsistent labels for {trial_id}")
            grouped[trial_id] = (total + value, n + 1, label)
        else:
            grouped[trial_id] = (value.copy(), 1, label)
    ids = list(grouped)
    return (
        np.stack([grouped[i][0] / grouped[i][1] for i in ids]),
        np.array([grouped[i][2] for i in ids]),
        ids,
    )


def metrics(predictions, labels, regression=False):
    if regression:
        return {
            "mse_percent_vv_squared": float(
                np.mean((predictions.reshape(-1) * 100 - labels * 100) ** 2)
            ),
            "trials": len(labels),
        }
    from sklearn.metrics import accuracy_score, f1_score

    pred = predictions.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "trials": len(labels),
    }


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
