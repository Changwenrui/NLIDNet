import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nlidnet.protocol import (
    NOVEL_LABELS,
    CONCENTRATIONS,
    support_indices,
    read_manifest,
)
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.config import save_config
from nlidnet.utils import (
    set_seed,
    build_model,
    move_inputs,
    aggregate_trials,
    metrics,
    save_json,
)


def select_support(records, shots, seed, order=None):
    if order is None:
        return [records[i] for i in support_indices(records, shots, seed)]
    with Path(order).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = {r["prefix"] for r in rows if int(r["rank"]) <= shots}
    selected = [r for r in records if r["prefix"] in chosen]
    expected = {r["label"] for r in records}
    if any(sum(r["label"] == label for r in selected) != shots for label in expected):
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
    return np.asarray(values), np.asarray(targets), ids


def adapt(cfg, checkpoint, task, method, shots, seed, support_order=None, epochs=None):
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    regression = task == "regression"
    labels = tuple(CONCENTRATIONS) if regression else NOVEL_LABELS
    rows = read_manifest(cfg["manifest"])
    train_rows = [r for r in rows if r["split"] == "train" and r["label"] in labels]
    test_rows = [r for r in rows if r["split"] == "test" and r["label"] in labels]
    if {r["label"] for r in train_rows} != set(labels) or {
        r["label"] for r in test_rows
    } != set(labels):
        raise ValueError("Every target label requires both training and test trials")
    selected = select_support(train_rows, shots, seed, support_order)
    budget = epochs or (500 if method == "linear_probe" and not regression else 100)
    out = Path(cfg["output"]) / cfg["stage"] / task / method / f"{shots}shot_seed{seed}"
    if (out / "final.pth").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    save_config(
        {
            **cfg,
            "task": task,
            "method": method,
            "shots": shots,
            "support_seed": seed,
            "source_epochs": cfg["epochs"],
            "epochs": budget,
            "source_checkpoint": checkpoint,
        },
        out,
    )
    with (out / "support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "split", "prefix", "label"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in selected)
    if method != "no_task_pretrain" and checkpoint is None:
        raise ValueError("A stage-matched source checkpoint is required")
    model = build_model(cfg, None if method == "no_task_pretrain" else checkpoint).to(
        cfg["device"]
    )
    for p in model.physics_head.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False
    head = nn.Linear(256, 1 if regression else len(labels)).to(cfg["device"])
    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    if method == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        x, y, _ = collect(
            model, None, cfg, selected, labels, "train", regression, features=True
        )
        cached = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=128,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        ds = LiquidEventDataset(cfg, "train", labels, offset=0, records=selected)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters() if p.requires_grad],
                    "lr": 2e-5 if method == "full_ft" else 1e-4,
                },
                {
                    "params": head.parameters(),
                    "lr": 2e-4 if method == "full_ft" else 1e-4,
                },
            ],
            weight_decay=1e-4,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=budget, eta_min=1e-6
    )
    for epoch in range(budget):
        head.train()
        if method == "linear_probe":
            batches = ((x.to(cfg["device"]), y.to(cfg["device"])) for x, y in cached)
        else:
            model.train()
            ds.offset = cfg["eval_offsets"][epoch % len(cfg["eval_offsets"])]
            loader = DataLoader(
                ds,
                batch_size=2,
                shuffle=True,
                num_workers=cfg["workers"],
                collate_fn=collate_trials,
            )
            batches = (
                (
                    model(*move_inputs(b, cfg["device"]))[2],
                    target(b, cfg["device"], regression),
                )
                for b in loader
            )
        total = 0.0
        count = 0
        for feat, y in batches:
            optimizer.zero_grad(set_to_none=True)
            pred = head(feat)
            loss = criterion(
                pred.flatten() if regression else pred,
                y.float() if regression else y.long(),
            )
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
            count += 1
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print({"epoch": epoch + 1, "loss": total / max(count, 1)}, flush=True)

    if method == "linear_probe":
        x, y, ids = collect(
            model, None, cfg, test_rows, labels, "test", regression, features=True
        )
        head.eval()
        with torch.no_grad():
            values = head(torch.from_numpy(x).to(cfg["device"])).cpu().numpy()
    else:
        values, y, ids = collect(
            model, head, cfg, test_rows, labels, "test", regression
        )
    values, y, ids = aggregate_trials(values, y.tolist(), ids)
    result = metrics(values, y, regression)
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "config": cfg,
            "task": task,
            "method": method,
            "shots": shots,
            "seed": seed,
            "epochs": budget,
        },
        out / "final.pth",
    )
    save_json(out / "metrics.json", result)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "target", "prediction"])
        pred = values.flatten() * 100 if regression else values.argmax(axis=1)
        w.writerows(zip(ids, y * 100 if regression else y, pred))
    print(result)
    return result
