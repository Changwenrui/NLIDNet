import csv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nlidnet.protocol import NOVEL_LABELS, read_manifest
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
import argparse
import json
from nlidnet.config import load_config
from nlidnet.training.support import select_support, target, collect


def add_config(parser):
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--device")
    return parser


def configuration(args):
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    return cfg


def run_experiment(
    cfg, checkpoint, method, shots, seed, support_order=None, epochs=None
):
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    labels = NOVEL_LABELS
    rows = read_manifest(cfg["manifest"])
    train_rows = [r for r in rows if r["split"] == "train" and r["label"] in labels]
    test_rows = [r for r in rows if r["split"] == "test" and r["label"] in labels]
    if {r["label"] for r in train_rows} != set(labels) or {
        r["label"] for r in test_rows
    } != set(labels):
        raise ValueError("Every target label requires both training and test trials")
    selected = select_support(train_rows, shots, seed, support_order)
    budget = epochs or (500 if method == "linear_probe" else 100)
    out = (
        Path(cfg["output"])
        / cfg["stage"]
        / "classification"
        / method
        / f"{shots}shot_seed{seed}"
    )
    if (out / "final.pth").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    save_config(
        {
            **cfg,
            "task": "classification",
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
        w.writerows(({k: r[k] for k in w.fieldnames} for r in selected))
    if method != "no_task_pretrain" and checkpoint is None:
        raise ValueError("A stage-matched source checkpoint is required")
    model = build_model(cfg, None if method == "no_task_pretrain" else checkpoint).to(
        cfg["device"]
    )
    for p in model.physics_head.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False
    head = nn.Linear(256, len(labels)).to(cfg["device"])
    criterion = nn.CrossEntropyLoss()
    if method == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        x, y, _ = collect(
            model, None, cfg, selected, labels, "train", False, features=True
        )
        cached = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=128,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    else:
        ds = LiquidEventDataset(cfg, "train", labels, offset=0, records=selected)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters() if p.requires_grad],
                    "lr": 2e-05 if method == "full_ft" else 0.0001,
                },
                {
                    "params": head.parameters(),
                    "lr": 0.0002 if method == "full_ft" else 0.0001,
                },
            ],
            weight_decay=0.0001,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=budget, eta_min=1e-06
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
                    target(b, cfg["device"], False),
                )
                for b in loader
            )
        total = 0.0
        count = 0
        for feat, y in batches:
            optimizer.zero_grad(set_to_none=True)
            pred = head(feat)
            loss = criterion(pred, y.long())
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
            count += 1
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print({"epoch": epoch + 1, "loss": total / max(count, 1)}, flush=True)
    if method == "linear_probe":
        x, y, ids = collect(
            model, None, cfg, test_rows, labels, "test", False, features=True
        )
        head.eval()
        with torch.no_grad():
            values = head(torch.from_numpy(x).to(cfg["device"])).cpu().numpy()
    else:
        values, y, ids = collect(model, head, cfg, test_rows, labels, "test", False)
    values, y, ids = aggregate_trials(values, y.tolist(), ids)
    result = metrics(values, y, False)
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "config": cfg,
            "task": "classification",
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
        pred = values.argmax(axis=1)
        w.writerows(zip(ids, y, pred))
    print(result)
    return result


def main(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint")
    p.add_argument(
        "--method",
        choices=["linear_probe", "full_ft", "no_task_pretrain"],
        default="linear_probe",
    )
    p.add_argument(
        "--shots", type=int, nargs="+", choices=[1, 5, 10, 20, 30, 40], default=[5]
    )
    p.add_argument("--seeds", "--seed", type=int, nargs="+", default=[1])
    p.add_argument("--support-order")
    p.add_argument("--epochs", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    cfg = configuration(args)
    if (
        args.method != "no_task_pretrain"
        and (not args.checkpoint)
        and (not args.dry_run)
    ):
        p.error("--checkpoint is required for this method")
    if args.dry_run:
        print(
            json.dumps(
                {
                    **cfg,
                    "task": "classification",
                    "method": args.method,
                    "shots": args.shots,
                    "seeds": args.seeds,
                },
                indent=2,
            )
        )
        return
    for seed in args.seeds:
        for shots in args.shots:
            run_experiment(
                cfg,
                args.checkpoint,
                args.method,
                shots,
                seed,
                args.support_order,
                args.epochs,
            )


if __name__ == "__main__":
    main()
