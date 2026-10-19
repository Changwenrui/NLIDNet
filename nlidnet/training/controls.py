import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from nlidnet.data.controls import ROISignalDataset, RGBFrameDataset
from nlidnet.models.baselines.temporal import build_temporal_model
from nlidnet.utils import aggregate_trials, metrics, save_json, set_seed
from nlidnet.config import save_config


def run_control(
    cfg,
    kind,
    model_name="convtran",
    task="classification",
    epochs=150,
    learning_rate=1e-4,
    weight_decay=1e-4,
    batch_size=8,
    checkpoint=None,
    initialization=None,
    video_root=None,
    frame_mode="first_frame",
    freeze_backbone=False,
    grayscale=False,
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
    regression = task == "regression"
    if kind == "temporal":
        model = build_temporal_model(
            model_name,
            channels=len(channels) if channels else 6,
            length=cfg["window_bins"],
            outputs=1 if regression else 16,
        )
        train_set = (
            ROISignalDataset(cfg, "train", task, channels=channels, normalize=normalize)
            if checkpoint is None
            else None
        )
        name = f"{model_name}_{task}_{cfg['stage']}"
    elif kind == "rgb":
        if regression:
            raise ValueError("RGB appearance controls use classification")
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
                cfg["manifest"],
                video_root,
                "train",
                frame_mode,
                seed,
                grayscale=grayscale,
            )
            if checkpoint is None
            else None
        )
        name = f"rgb_{frame_mode}_{'frozen' if freeze_backbone else 'finetuned'}_{'gray' if grayscale else 'color'}"
    else:
        raise ValueError(f"Unknown control kind: {kind}")
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
            "kind": kind,
            "model_name": model_name,
            "task": task,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "frame_mode": frame_mode,
            "freeze_backbone": freeze_backbone,
            "grayscale": grayscale,
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
            optimizer, T_max=epochs, eta_min=1e-6
        )
        with (out / "train.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "loss"])
            for epoch in range(epochs):
                model.train()
                if kind == "rgb" and freeze_backbone:
                    model.eval()
                    model.fc.train()
                total, count = 0.0, 0
                for x, y, _ in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x.to(cfg["device"]))
                    target = y.to(cfg["device"])
                    loss = (
                        nn.functional.mse_loss(logits.reshape(-1), target.float())
                        if regression
                        else nn.functional.cross_entropy(logits, target.long())
                    )
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
    scores, targets, ids = [], [], []
    offsets = cfg["eval_offsets"] if kind == "temporal" else [0]
    with torch.no_grad():
        for offset in offsets:
            dataset = (
                ROISignalDataset(
                    cfg,
                    "test",
                    task,
                    offset=offset,
                    channels=channels,
                    normalize=normalize,
                )
                if kind == "temporal"
                else RGBFrameDataset(
                    cfg["manifest"],
                    video_root,
                    "test",
                    frame_mode,
                    seed,
                    grayscale=grayscale,
                )
            )
            for x, y, trials in DataLoader(
                dataset, batch_size=batch_size, num_workers=cfg["workers"]
            ):
                scores.extend(model(x.to(cfg["device"])).cpu().numpy())
                targets.extend(y.tolist())
                ids.extend(trials)
    if regression:
        values = np.asarray(scores).reshape(-1)
        groups = {}
        for value, target, trial in zip(values, targets, ids):
            if trial not in groups:
                groups[trial] = [[], target]
            groups[trial][0].append(value)
        ids = sorted(groups)
        values = np.asarray([np.mean(groups[trial][0]) for trial in ids])
        targets = np.asarray([groups[trial][1] for trial in ids])
        result = {
            "mse_percentage_points": float(np.mean(((values - targets) * 100) ** 2)),
            "trials": len(ids),
        }
        predictions = values * 100
        targets = targets * 100
    else:
        values, targets, ids = aggregate_trials(np.asarray(scores), targets, ids)
        result = metrics(values, targets)
        predictions = values.argmax(axis=1)
    save_json(out / "metrics.json", result)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "target", "prediction"])
        writer.writerows(zip(ids, targets, predictions))
    return result
