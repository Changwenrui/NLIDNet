import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.protocol import BASE_LABELS, NOVEL_LABELS, support_indices
from nlidnet.utils import aggregate_trials, metrics, save_json


def event_statistics(events, mask, delta_t_us=2000, bins=700):
    ev = events[mask.bool()].double().cpu().numpy()
    output = np.zeros(18, dtype=np.float64)
    if not len(ev):
        return output
    x, y, t, p = ev.T
    counts = np.bincount(
        np.clip((t / delta_t_us).astype(int), 0, bins - 1), minlength=bins
    )
    output[:8] = [
        len(ev),
        np.mean(p),
        np.mean(counts),
        np.std(counts),
        np.max(counts),
        np.count_nonzero(counts) / bins,
        np.quantile(counts, 0.25),
        np.quantile(counts, 0.75),
    ]
    output[8:] = [
        np.mean(x),
        np.std(x),
        np.mean(y),
        np.std(y),
        np.mean(t),
        np.std(t),
        np.quantile(x, 0.1),
        np.quantile(x, 0.9),
        np.quantile(y, 0.1),
        np.quantile(y, 0.9),
    ]
    return output


def classify(cfg, output, shots=None, seed=1):
    labels = NOVEL_LABELS if shots is not None else BASE_LABELS
    train = LiquidEventDataset(cfg, "train", labels, offset=0)
    if shots is not None:
        chosen = support_indices(train.records, shots, seed)
        rows = [train.records[i] for i in chosen]
        train = LiquidEventDataset(cfg, "train", labels, offset=0, records=rows)

    def collect(split, records=None):
        features = []
        targets = []
        ids = []
        for offset in cfg["eval_offsets"]:
            ds = LiquidEventDataset(cfg, split, labels, offset=offset, records=records)
            for batch in DataLoader(
                ds, batch_size=1, num_workers=cfg["workers"], collate_fn=collate_trials
            ):
                features.append(
                    event_statistics(
                        batch["inputs"][1][0],
                        batch["inputs"][2][0],
                        cfg["delta_t_us"],
                        cfg["window_bins"],
                    )
                )
                targets.append(int(batch["labels"][0]))
                ids.extend(batch["trial_ids"])
        return np.asarray(features), np.asarray(targets), ids

    x, y, _ = collect("train", train.records)
    if len(set(y)) < 2:
        raise ValueError("At least two training classes are required")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    )
    model.fit(x, y)
    x, y, ids = collect("test")
    probabilities = np.zeros((len(x), len(labels)))
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(x)
    values, y, ids = aggregate_trials(probabilities, y.tolist(), ids)
    result = metrics(values, y)
    save_json(output, result)
    return result
