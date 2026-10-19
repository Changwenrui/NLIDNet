from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.utils import build_model, move_inputs, save_json
import argparse
from nlidnet.config import load_config


def benchmark(
    cfg, checkpoint, output, batches=50, warmup=5, batch_size=1, precision="float32"
):
    if batches < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("Invalid benchmark budget")
    device = torch.device(cfg["device"])
    model = build_model(cfg, checkpoint).to(device).eval()
    data = LiquidEventDataset(cfg, "test", offset=0)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=cfg["workers"],
        collate_fn=collate_trials,
    )
    iterator = iter(loader)
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step in range(warmup + batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            inputs = move_inputs(batch, device)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and precision == "bfloat16"
                else nullcontext()
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            with context:
                model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (perf_counter() - start) * 1000
            if step >= warmup:
                rows.append(
                    {
                        "batch": step - warmup,
                        "views": len(batch["trial_ids"]),
                        "milliseconds": elapsed,
                    }
                )
    times = np.asarray([r["milliseconds"] for r in rows])
    views = sum((r["views"] for r in rows))
    result = {
        "device": str(device),
        "precision": precision,
        "batches": batches,
        "mean_batch_ms": float(times.mean()),
        "median_batch_ms": float(np.median(times)),
        "mean_view_ms": float(times.sum() / views),
        "views_per_second": float(views / (times.sum() / 1000)),
    }
    if device.type == "cuda":
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "timing.json", result)
    with (out / "batches.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return result


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


def main(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--output", default="outputs/inference_speed")
    args = p.parse_args(argv)
    benchmark(
        configuration(args),
        args.checkpoint,
        args.output,
        args.batches,
        args.warmup,
        args.batch_size,
        args.precision,
    )


if __name__ == "__main__":
    main()
