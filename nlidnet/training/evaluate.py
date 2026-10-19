import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.utils import build_model, move_inputs, aggregate_trials, metrics, save_json


@torch.no_grad()
def evaluate(cfg, checkpoint):
    model = build_model(cfg, checkpoint).to(cfg["device"]).eval()
    values = []
    labels = []
    ids = []
    for offset in cfg["eval_offsets"]:
        ds = LiquidEventDataset(cfg, "test", offset=offset)
        loader = DataLoader(
            ds, batch_size=1, num_workers=cfg["workers"], collate_fn=collate_trials
        )
        for batch in loader:
            logits = model(*move_inputs(batch, cfg["device"]))[0]
            values.extend(logits.cpu().numpy())
            labels.extend(batch["labels"].tolist())
            ids.extend(batch["trial_ids"])
    scores, labels, ids = aggregate_trials(np.asarray(values), labels, ids)
    result = metrics(scores, labels)
    out = Path(cfg["output"]) / cfg["stage"] / "evaluation"
    save_json(out / "metrics.json", result)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "label", "prediction"])
        w.writerows(zip(ids, labels, scores.argmax(axis=1)))
    print(result)
    return result
