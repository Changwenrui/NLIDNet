import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from nlidnet.data.controls import RGBFrameDataset
from nlidnet.utils import aggregate_trials, metrics, save_json, set_seed
from nlidnet.config import save_config
import argparse
from nlidnet.config import load_config


def run_experiment(
    cfg,
    epochs=150,
    learning_rate=0.0001,
    weight_decay=0.0001,
    batch_size=8,
    checkpoint=None,
    initialization=None,
    video_root=None,
    frame_mode="first_frame",
    freeze_backbone=False,
    grayscale=False,
    seed=42,
    output="outputs/controls",
):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid control training budget")
    set_seed(seed)
    from torchvision import models

    model = models.resnet34(weights=None)
    if checkpoint is None:
        if initialization is None:
            raise ValueError("RGB controls require ImageNet initialization weights")
        state = torch.load(initialization, map_location="cpu", weights_only=True)
        state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}
        )
    model.fc = nn.Linear(model.fc.in_features, 16)
    if freeze_backbone:
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    train_set = (
        RGBFrameDataset(
            cfg["manifest"], video_root, "train", frame_mode, seed, grayscale=grayscale
        )
        if checkpoint is None
        else None
    )
    name = f"rgb_{frame_mode}_{('frozen' if freeze_backbone else 'finetuned')}_{('gray' if grayscale else 'color')}"
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
            "kind": "rgb",
            "model_name": "resnet34",
            "task": "classification",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "frame_mode": frame_mode,
            "freeze_backbone": freeze_backbone,
            "grayscale": grayscale,
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
                if freeze_backbone:
                    model.eval()
                    model.fc.train()
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
    offsets = [0]
    with torch.no_grad():
        for offset in offsets:
            dataset = RGBFrameDataset(
                cfg["manifest"],
                video_root,
                "test",
                frame_mode,
                seed,
                grayscale=grayscale,
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
    parser = argparse.ArgumentParser(description="rgb classification")
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
    parser.add_argument("--video-root", default="data/raw")
    parser.add_argument(
        "--initialization", default="checkpoints/resnet34_imagenet1k_v1.pth"
    )
    parser.add_argument(
        "--frame-mode",
        choices=["first_frame", "last_frame", "random_pour"],
        default="first_frame",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
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
        video_root=args.video_root,
        initialization=args.initialization,
        frame_mode=args.frame_mode,
        freeze_backbone=args.freeze_backbone,
        grayscale=args.grayscale,
    )


if __name__ == "__main__":
    main()
