import argparse
import json
from nlidnet.config import load_config


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
    from nlidnet.training.train import train

    return train(cfg)


if __name__ == "__main__":
    main()
