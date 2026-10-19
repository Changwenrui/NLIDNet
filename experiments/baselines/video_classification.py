import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from nlidnet.models.baselines import build_video_model
from nlidnet.models.baselines.data import BaselineDataset
from nlidnet.utils import set_seed
from nlidnet.protocol import STAGES
import csv
import numpy as np
from nlidnet.utils import metrics, save_json


def train(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["videomae_v2", "go_with_the_flow", "dont_look_twice"],
        required=True,
    )
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output", default="outputs/baselines")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    set_seed(42)
    epochs, lr, wd, step = (150, 8e-06, 0.001, 30)
    ds = BaselineDataset(
        args.manifest, args.data_root, "train", args.model, STAGES[args.stage] * 0.002
    )
    if len(ds) < 8:
        raise ValueError(
            "At least 8 training trials are required for the baseline batch size"
        )
    loader = DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=args.workers, drop_last=True
    )
    model = build_video_model(args.model, checkpoint_dir=args.checkpoint_dir).to(
        args.device
    )
    out = Path(args.output) / args.model / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pth").exists():
        raise FileExistsError(out / "final.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.7)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(args.device))
            loss = nn.functional.cross_entropy(logits, y.to(args.device))
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
            if args.model == "spiking_resformer":
                from spikingjelly.activation_based import functional

                functional.reset_net(model)
        schedule.step()
        print({"epoch": epoch + 1, "loss": total / len(loader)}, flush=True)
    torch.save(model.state_dict(), out / "final.pth")


def evaluate(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["videomae_v2", "go_with_the_flow", "dont_look_twice"],
        required=True,
    )
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/baseline_evaluation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    ds = BaselineDataset(
        args.manifest, args.data_root, "test", args.model, STAGES[args.stage] * 0.002
    )
    model = build_video_model(args.model, pretrained=False)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.to(args.device).eval()
    scores = []
    labels = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=args.workers):
            scores.extend(model(x.to(args.device)).cpu().numpy())
            labels.extend(y.tolist())
            if args.model == "spiking_resformer":
                from spikingjelly.activation_based import functional

                functional.reset_net(model)
    result = metrics(np.asarray(scores), np.asarray(labels))
    out = Path(args.output) / args.model / args.stage
    save_json(out / "metrics.json", result)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "label", "prediction"])
        w.writerows(
            zip(
                [r["trial_id"] for r in ds.rows],
                labels,
                np.asarray(scores).argmax(axis=1),
            )
        )
    print(result)


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate"])
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return
    action = args.pop(0)
    if action == "train":
        return train(args)
    if action == "evaluate":
        return evaluate(args)
    parser.error("Choose train or evaluate")


if __name__ == "__main__":
    main()
