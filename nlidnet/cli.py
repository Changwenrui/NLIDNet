import argparse
import json
from nlidnet.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="nlidnet", description="Event-based liquid response encoding"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("preprocess", help="Convert manifest-indexed RAW trials")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--raw-root", required=True)
    prep.add_argument("--output", required=True)
    check = commands.add_parser(
        "check-data", help="Validate manifest and tensor pairing"
    )
    train = commands.add_parser("train", help="Train the 16-class source model")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate source classification at trial level"
    )
    evaluate.add_argument("--checkpoint", required=True)
    adapt = commands.add_parser(
        "adapt", help="Few-shot classification or concentration regression"
    )
    adapt.add_argument("--checkpoint")
    adapt.add_argument(
        "--task", choices=["classification", "regression"], required=True
    )
    adapt.add_argument(
        "--method",
        choices=["linear_probe", "full_ft", "no_task_pretrain"],
        required=True,
    )
    adapt.add_argument("--shots", type=int, choices=[1, 5, 10, 20], required=True)
    adapt.add_argument("--seed", type=int, default=1)
    adapt.add_argument(
        "--support-order", help="Support-order CSV with rank and prefix columns"
    )
    adapt.add_argument(
        "--epochs", type=int, help="Explicit experimental override of the paper budget"
    )
    export = commands.add_parser(
        "export", help="Export per-trial embeddings and a PCA plot"
    )
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--split", choices=["train", "test"], default="test")
    visualize = commands.add_parser(
        "visualize", help="Inspect temporal attention or Grad-CAM for one test trial"
    )
    visualize.add_argument("--checkpoint", required=True)
    visualize.add_argument("--output", required=True)
    visualize.add_argument("--trial-id", required=True)
    visualize.add_argument(
        "--kind", choices=["attention", "gradcam"], default="attention"
    )
    for cmd in [check, train, evaluate, adapt, export, visualize]:
        cmd.add_argument("--config", default="configs/paper.yaml")
        cmd.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
        cmd.add_argument("--device", help="cuda, cuda:0, or cpu")
        cmd.add_argument(
            "--dry-run",
            action="store_true",
            help="Print resolved configuration without importing PyTorch",
        )
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        from nlidnet.data.preprocessing import preprocess

        return preprocess(args.manifest, args.raw_root, args.output)
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    if args.command == "check-data":
        from nlidnet.data.validate import validate_data

        return validate_data(cfg)
    if args.command == "train":
        from nlidnet.training.train import train

        train(cfg)
        return
    if args.command == "evaluate":
        from nlidnet.training.evaluate import evaluate

        evaluate(cfg, args.checkpoint)
        return
    if args.command == "export":
        from nlidnet.visualization.export import export_features

        export_features(cfg, args.checkpoint, args.output, args.split)
        return
    if args.command == "visualize":
        from nlidnet.visualization.export import inspect_trial

        inspect_trial(cfg, args.checkpoint, args.trial_id, args.output, args.kind)
        return
    from nlidnet.training.adapt import adapt

    adapt(
        cfg,
        args.checkpoint,
        args.task,
        args.method,
        args.shots,
        args.seed,
        args.support_order,
        args.epochs,
    )
