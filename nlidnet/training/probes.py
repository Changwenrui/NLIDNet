import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nlidnet.analysis.features import load_features
from nlidnet.protocol import NOVEL_LABELS, CONCENTRATIONS, support_indices
from nlidnet.utils import set_seed, aggregate_trials, metrics, save_json


def run_probe(
    train_path,
    test_path,
    output,
    task="classification",
    head="linear",
    shots=5,
    seed=1,
    epochs=None,
    dropout=0.2,
    device="cuda",
    normalize=True,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    label_smoothing=0.0,
):
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    regression = task == "regression"
    label_order = tuple(CONCENTRATIONS) if regression else NOVEL_LABELS
    label_map = {label: i for i, label in enumerate(label_order)}
    train = load_features(train_path)
    test = load_features(test_path)
    if train.features.shape[1] != test.features.shape[1]:
        raise ValueError("Feature dimensions differ")
    if set(train.trial_ids) & set(test.trial_ids):
        raise ValueError("Training and test trials overlap")
    records = []
    seen = {}
    for label, trial in zip(train.labels, train.trial_ids):
        if label not in label_map:
            continue
        if trial in seen:
            if seen[trial] != label:
                raise ValueError("Inconsistent labels for a support trial")
            continue
        seen[trial] = int(label)
        records.append({"trial_id": str(trial), "split": "train", "label": int(label)})
    if {r["label"] for r in records} != set(label_order):
        raise ValueError("Missing target classes in training features")
    selected = [records[i] for i in support_indices(records, shots, seed)]
    selected_ids = {r["trial_id"] for r in selected}
    train_mask = np.isin(train.trial_ids, list(selected_ids))
    test_mask = np.isin(test.labels, label_order)
    if set(test.labels[test_mask]) != set(label_order):
        raise ValueError("Missing target classes in test features")

    def target(labels):
        return np.asarray(
            [
                CONCENTRATIONS[int(v)] / 100 if regression else label_map[int(v)]
                for v in labels
            ]
        )

    x = torch.from_numpy(train.features[train_mask])
    y = torch.from_numpy(target(train.labels[train_mask]))
    query = torch.from_numpy(test.features[test_mask])
    if normalize:
        x = nn.functional.normalize(x, dim=1)
        query = nn.functional.normalize(query, dim=1)
    set_seed(seed)
    dimension = x.shape[1]
    classes = 1 if regression else len(label_order)
    network = (
        nn.Linear(dimension, classes)
        if head == "linear"
        else nn.Sequential(
            nn.Linear(dimension, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, classes),
        )
    )
    network.to(device)
    budget = epochs or (100 if regression else 500)
    if budget < 1:
        raise ValueError("epochs must be positive")
    out = Path(output) / head / f"{shots}shot_seed{seed}"
    if (out / "head.pth").exists():
        raise FileExistsError(out / "head.pth")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "split", "label"])
        w.writeheader()
        w.writerows(selected)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
    criterion = (
        nn.MSELoss()
        if regression
        else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    )
    for epoch in range(budget):
        network.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = network(features.to(device))
            loss = criterion(
                pred.flatten() if regression else pred,
                labels.to(device).float() if regression else labels.to(device).long(),
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
    network.eval()
    with torch.no_grad():
        values = np.concatenate(
            [network(chunk.to(device)).cpu().numpy() for chunk in query.split(128)]
        )
    values, labels, ids = aggregate_trials(
        values, target(test.labels[test_mask]).tolist(), test.trial_ids[test_mask]
    )
    result = metrics(values, labels, regression)
    metadata = {
        "task": task,
        "head": head,
        "shots": shots,
        "seed": seed,
        "epochs": budget,
        "normalize": normalize,
        "input_dim": dimension,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "label_smoothing": label_smoothing,
    }
    torch.save(
        {"head": network.state_dict(), "config": metadata, "labels": label_order},
        out / "head.pth",
    )
    save_json(out / "config.json", metadata)
    save_json(out / "metrics.json", result)
    return result
