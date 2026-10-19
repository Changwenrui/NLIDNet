import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from nlidnet.data.controls import ROISignalDataset
from nlidnet.models.baselines.temporal import build_temporal_model
from nlidnet.utils import aggregate_trials, metrics, save_json, set_seed
from nlidnet.config import save_config
import argparse
from nlidnet.config import load_config


def run_experiment(
    cfg,
    model_name="convtran",
    epochs=150,
    learning_rate=0.0001,
    weight_decay=0.0001,
    batch_size=8,
    checkpoint=None,
    channels=None,
    normalize=False,
    seed=42,
    output="outputs/controls",
):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid control training budget")
    if channels is not None and (
        not channels or min(channels) < 0 or max(channels) > 5
    ):
        raise ValueError("ROI channel indices must lie in 0..5")
    set_seed(seed)
    model = build_temporal_model(
        model_name,
        channels=len(channels) if channels else 6,
        length=cfg["window_bins"],
        outputs=16,
    )
    train_set = (
        ROISignalDataset(
            cfg, "train", "classification", channels=channels, normalize=normalize
        )
        if checkpoint is None
        else None
    )
    name = f"{model_name}_classification_{cfg['stage']}"
    out = Path(output) / name / f"seed_{seed}"
    model.to(cfg["device"])
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    else:
        if (out / "final.pth").exists():
            raise FileExistsError(out / "final.pth")
        settings = {
            **cfg,
            "kind": "temporal",
            "model_name": model_name,
            "task": "classification",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "channels": channels,
            "normalize": normalize,
            "seed": seed,
        }
        save_config(settings, out)
        loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=cfg["workers"]
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-06
        )
        with (out / "train.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "loss"])
            for epoch in range(epochs):
                model.train()
                total, count = (0.0, 0)
                for x, y, _ in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x.to(cfg["device"]))
                    target = y.to(cfg["device"])
                    loss = nn.functional.cross_entropy(logits, target.long())
                    loss.backward()
                    optimizer.step()
                    total += loss.detach().item() * len(x)
                    count += len(x)
                scheduler.step()
                writer.writerow([epoch + 1, total / count])
                stream.flush()
                print({"epoch": epoch + 1, "loss": total / count}, flush=True)
        torch.save({"model": model.state_dict(), "config": settings}, out / "final.pth")
    model.eval()
    scores, targets, ids = ([], [], [])
    offsets = cfg["eval_offsets"]
    with torch.no_grad():
        for offset in offsets:
            dataset = ROISignalDataset(
                cfg,
                "test",
                "classification",
                offset=offset,
                channels=channels,
                normalize=normalize,
            )
            for x, y, trials in DataLoader(
                dataset, batch_size=batch_size, num_workers=cfg["workers"]
            ):
                scores.extend(model(x.to(cfg["device"])).cpu().numpy())
                targets.extend(y.tolist())
                ids.extend(trials)
    values, targets, ids = aggregate_trials(np.asarray(scores), targets, ids)
    result = metrics(values, targets)
    predictions = values.argmax(axis=1)
    save_json(out / "metrics.json", result)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "target", "prediction"])
        writer.writerows(zip(ids, targets, predictions))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="temporal classification")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument(
        "--stage", choices=["stage1", "stage2", "stage3"], default="stage1"
    )
    parser.add_argument("--device")
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="outputs/controls")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=["convtran", "transformer", "inception_time"],
        default="convtran",
    )
    parser.add_argument("--channels", type=int, nargs="+")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    if args.manifest:
        cfg["manifest"] = args.manifest
    return run_experiment(
        cfg,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        seed=args.seed,
        output=args.output,
        model_name=args.model,
        channels=args.channels,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
