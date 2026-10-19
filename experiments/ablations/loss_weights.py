import argparse
from itertools import product
from pathlib import Path
import json
from experiments._arguments import add_config, configuration


def main():
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--variance", type=float, nargs="+", default=[0, 0.0005])
    p.add_argument("--contrastive", type=float, nargs="+", default=[0, 0.02])
    p.add_argument("--physics", type=float, nargs="+", default=[0, 0.10])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--epochs", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    base = configuration(args)
    if args.epochs is not None:
        if args.epochs < 1:
            p.error("epochs must be positive")
        base["epochs"] = args.epochs
    for variance, contrastive, physics, seed in product(
        args.variance, args.contrastive, args.physics, args.seeds
    ):
        if min(variance, contrastive, physics) < 0:
            p.error("Loss weights must be nonnegative")
        name = f"variance_{variance:g}_contrastive_{contrastive:g}_physics_{physics:g}_seed_{seed}"
        cfg = {
            **base,
            "lambda_var": variance,
            "lambda_supcon": contrastive,
            "lambda_phys": physics,
            "seed": seed,
            "output": str(Path(base["output"]) / "loss_weights" / name),
        }
        if args.dry_run:
            print(json.dumps(cfg, indent=2))
        else:
            from nlidnet.training.train import train

            train(cfg)


if __name__ == "__main__":
    main()
