import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from nlidnet.config import save_config
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.data.sampling import ClassBalancedBatchSampler
from nlidnet.utils import build_model, set_seed, move_inputs
from nlidnet.training.losses import LiquidRepresentationLoss
import argparse
import json
from nlidnet.config import load_config


def run_experiment(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    out = Path(cfg["output"]) / cfg["stage"] / "pretrain"
    if (out / "final.pth").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    save_config(cfg, out)
    ds = LiquidEventDataset(cfg, "train")
    sampler = ClassBalancedBatchSampler(
        ds, cfg["classes_per_batch"], cfg["samples_per_class"], seed=cfg["seed"]
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, num_workers=cfg["workers"], collate_fn=collate_trials
    )
    model = build_model(cfg).to(device)
    criterion = LiquidRepresentationLoss(
        cfg["T"], cfg["k"], cfg["lambda_var"], cfg["lambda_supcon"], cfg["lambda_phys"]
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
    )
    with (out / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "loss", "ce", "variance", "supcon", "physics"]
        )
        writer.writeheader()
        for epoch in range(cfg["epochs"]):
            sampler.set_epoch(epoch)
            model.train()
            sums = {key: 0.0 for key in ["loss", "ce", "variance", "supcon", "physics"]}
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    terms = criterion(
                        model(*move_inputs(batch, device)),
                        batch["labels"].to(device),
                        batch["physics"].to(device),
                    )
                    loss = terms["loss"]
                loss.backward()
                optimizer.step()
                for key in sums:
                    sums[key] += terms[key].detach().item()
            scheduler.step()
            row = {"epoch": epoch + 1, **{k: v / len(loader) for k, v in sums.items()}}
            writer.writerow(row)
            f.flush()
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "config": cfg},
                out / "latest.pth",
            )
            print(row, flush=True)
        torch.save(
            {"model": model.state_dict(), "epoch": cfg["epochs"], "config": cfg},
            out / "final.pth",
        )
    return out / "final.pth"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train NLIDNet on source liquid categories"
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    for key in ("epochs", "learning_rate", "manifest", "output"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if cfg["epochs"] < 1 or cfg["learning_rate"] <= 0:
        parser.error("epochs and learning rate must be positive")
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
