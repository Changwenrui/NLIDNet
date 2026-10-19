import argparse
import json
from nlidnet.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate NLIDNet at independent trial level"
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    for key in ("manifest", "output"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    from nlidnet.training.evaluate import evaluate

    return evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
