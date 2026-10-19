import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.utils import build_model, move_inputs, aggregate_trials, metrics, save_json
import argparse
import json
from nlidnet.config import load_config


@torch.no_grad()
def run_experiment(cfg, checkpoint):
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate NLIDNet at independent trial level"
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    for key in ("manifest", "output"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    return run_experiment(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
