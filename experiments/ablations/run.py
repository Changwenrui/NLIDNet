import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from nlidnet.config import load_config, save_config
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.data.sampling import ClassBalancedBatchSampler
from nlidnet.models.ablations import NoFlowAblation, UniformTemporalAblation
from nlidnet.models.sparse_sampling import SparseTemporalAblation
from nlidnet.utils import (
    set_seed,
    move_inputs,
    aggregate_trials,
    metrics,
    save_json,
)
from nlidnet.training.losses import calculate_variance_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation", choices=["no_flow", "uniform", "sparse"], required=True
    )
    p.add_argument("--sample-count", type=int, choices=[20, 50, 100], default=20)
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--epochs", type=int)
    p.add_argument("--sparse-bin-us", type=int, choices=[20, 200], default=200)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--train-offset-count", type=int)
    p.add_argument("--learning-rate", type=float)
    args = p.parse_args()
    cfg = load_config(args.config, args.stage)
    set_seed(cfg["seed"])
    if args.accumulation_steps < 1:
        p.error("accumulation-steps must be positive")
    if args.train_offset_count is not None:
        if args.train_offset_count < 1:
            p.error("train-offset-count must be positive")
        cfg["train_offset_count"] = args.train_offset_count
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            p.error("learning-rate must be positive")
        cfg["learning_rate"] = args.learning_rate
    cfg["accumulation_steps"] = args.accumulation_steps
    cfg["sparse_bin_us"] = args.sparse_bin_us
    if args.epochs is not None:
        if args.epochs < 1:
            p.error("epochs must be positive")
        cfg["epochs"] = args.epochs
    if args.evaluate and not args.checkpoint:
        p.error("--evaluate requires --checkpoint")
    init = {}
    if not args.checkpoint:
        for key in (
            ["resnet_checkpoint"]
            if args.ablation == "no_flow"
            else ["resnet_checkpoint", "flow_checkpoint"]
        ):
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(cfg[key])
            init[key] = cfg[key]
    if args.ablation == "no_flow":
        model = NoFlowAblation(
            T=cfg["T"], k=cfg["k"], delta_t=cfg["delta_t_us"], **init
        )
    elif args.ablation == "uniform":
        model = UniformTemporalAblation(
            sample_count=args.sample_count, delta_t=cfg["delta_t_us"], **init
        )
    else:
        model = SparseTemporalAblation(
            sample_count=args.sample_count,
            sparse_bin_us=args.sparse_bin_us,
            delta_t=cfg["delta_t_us"],
            **init,
        )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)
    model.to(cfg["device"])
    name = (
        "no_flow"
        if args.ablation == "no_flow"
        else f"{args.ablation}_{args.sample_count}frames"
    )
    if args.ablation == "sparse":
        name += f"_{args.sparse_bin_us}us"
    out = Path(cfg["output"]) / "ablations" / name / args.stage

    def forward(batch):
        x, ev, mask, offset = move_inputs(batch, cfg["device"])
        return (
            model(x, ev, mask, offset)
            if args.ablation == "no_flow"
            else (model(ev, mask, offset), None)
        )

    if not args.evaluate:
        if (out / "final.pth").exists():
            raise FileExistsError(out / "final.pth")
        save_config(
            {**cfg, "ablation": args.ablation, "sample_count": args.sample_count}, out
        )
        ds = LiquidEventDataset(cfg, "train")
        sampler = ClassBalancedBatchSampler(
            ds, cfg["classes_per_batch"], cfg["samples_per_class"], seed=cfg["seed"]
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=cfg["workers"],
            collate_fn=collate_trials,
        )
        optimizer = torch.optim.AdamW(
            [v for v in model.parameters() if v.requires_grad],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
        )
        for epoch in range(cfg["epochs"]):
            sampler.set_epoch(epoch)
            model.train()
            total = 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(loader):
                logits, attention = forward(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["labels"].to(cfg["device"])
                )
                if attention is not None:
                    loss = loss + cfg["lambda_var"] * calculate_variance_loss(
                        attention, cfg["T"], cfg["k"], torch.device(cfg["device"])
                    )
                group_start = (
                    batch_index // args.accumulation_steps
                ) * args.accumulation_steps
                group_size = min(args.accumulation_steps, len(loader) - group_start)
                (loss / group_size).backward()
                if (
                    batch_index + 1
                ) % args.accumulation_steps == 0 or batch_index + 1 == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.detach().item()
            scheduler.step()
            print({"epoch": epoch + 1, "loss": total / len(loader)}, flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "ablation": args.ablation,
                "sample_count": args.sample_count,
            },
            out / "final.pth",
        )
    else:
        model.eval()
        values = []
        targets = []
        ids = []
        with torch.no_grad():
            for offset in cfg["eval_offsets"]:
                ds = LiquidEventDataset(cfg, "test", offset=offset)
                for batch in DataLoader(
                    ds,
                    batch_size=1,
                    num_workers=cfg["workers"],
                    collate_fn=collate_trials,
                ):
                    values.extend(forward(batch)[0].cpu().numpy())
                    targets.extend(batch["labels"].tolist())
                    ids.extend(batch["trial_ids"])
        values, targets, ids = aggregate_trials(np.asarray(values), targets, ids)
        result = metrics(values, targets)
        save_json(out / "metrics.json", result)
        print(result)


if __name__ == "__main__":
    main()
