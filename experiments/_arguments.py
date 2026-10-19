import argparse
import json
from nlidnet.config import load_config


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


def adaptation(task):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint")
    p.add_argument(
        "--method",
        choices=["linear_probe", "full_ft", "no_task_pretrain"],
        default="linear_probe",
    )
    p.add_argument("--shots", type=int, nargs="+", choices=[1, 5, 10, 20], default=[5])
    p.add_argument("--seeds", "--seed", type=int, nargs="+", default=[1])
    p.add_argument("--support-order")
    p.add_argument("--epochs", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = configuration(args)
    if args.method != "no_task_pretrain" and not args.checkpoint and not args.dry_run:
        p.error("--checkpoint is required for this method")
    if args.dry_run:
        print(
            json.dumps(
                {
                    **cfg,
                    "task": task,
                    "method": args.method,
                    "shots": args.shots,
                    "seeds": args.seeds,
                },
                indent=2,
            )
        )
        return
    from nlidnet.training.adapt import adapt

    for seed in args.seeds:
        for shots in args.shots:
            adapt(
                cfg,
                args.checkpoint,
                task,
                args.method,
                shots,
                seed,
                args.support_order,
                args.epochs,
            )


def probes(task):
    p = argparse.ArgumentParser()
    p.add_argument("--train-features", required=True)
    p.add_argument("--test-features", required=True)
    p.add_argument("--heads", nargs="+", choices=["linear", "mlp"], default=["linear"])
    p.add_argument(
        "--shots", nargs="+", type=int, choices=[1, 5, 10, 20], default=[1, 5, 10, 20]
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--epochs", type=int)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--output",
        default=(
            "outputs/few_shot_probes"
            if task == "classification"
            else "outputs/concentration_probes"
        ),
    )
    args = p.parse_args()
    from nlidnet.training.probes import run_probe

    for seed in args.seeds:
        for shots in args.shots:
            for head in args.heads:
                run_probe(
                    args.train_features,
                    args.test_features,
                    args.output,
                    task,
                    head,
                    shots,
                    seed,
                    args.epochs,
                    args.dropout,
                    args.device,
                    not args.no_normalize,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    label_smoothing=args.label_smoothing,
                )
