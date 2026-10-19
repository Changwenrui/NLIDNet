import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from nlidnet.protocol import CONCENTRATIONS, support_indices
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.utils import move_inputs


def select_support(records, shots, seed, order=None):
    if order is None:
        return [records[i] for i in support_indices(records, shots, seed)]
    with Path(order).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = {r["prefix"] for r in rows if int(r["rank"]) <= shots}
    selected = [r for r in records if r["prefix"] in chosen]
    expected = {r["label"] for r in records}
    if any(
        (sum((r["label"] == label for r in selected)) != shots for label in expected)
    ):
        raise ValueError(
            "Support-order CSV does not provide exactly K available training trials per class"
        )
    return selected


def target(batch, device, regression):
    if regression:
        return torch.tensor(
            [CONCENTRATIONS[int(v)] / 100 for v in batch["raw_labels"]], device=device
        ).float()
    return batch["labels"].to(device)


def collect(model, head, cfg, records, labels, split, regression, features=False):
    values = []
    targets = []
    ids = []
    model.eval()
    if head is not None:
        head.eval()
    with torch.no_grad():
        for offset in cfg["eval_offsets"]:
            ds = LiquidEventDataset(cfg, split, labels, offset=offset, records=records)
            for batch in DataLoader(
                ds, batch_size=1, num_workers=cfg["workers"], collate_fn=collate_trials
            ):
                feat = model(*move_inputs(batch, cfg["device"]))[2]
                if features:
                    value = nn.functional.normalize(feat, dim=1)
                else:
                    value = head(feat)
                values.extend(value.cpu().numpy())
                targets.extend(target(batch, "cpu", regression).tolist())
                ids.extend(batch["trial_ids"])
    return (np.asarray(values), np.asarray(targets), ids)
