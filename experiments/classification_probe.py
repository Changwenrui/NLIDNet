import argparse


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train-features", required=True)
    p.add_argument("--test-features", required=True)
    p.add_argument("--heads", nargs="+", choices=["linear", "mlp"], default=["linear"])
    p.add_argument(
        "--shots",
        nargs="+",
        type=int,
        choices=[1, 5, 10, 20, 30, 40],
        default=[1, 5, 10, 20],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--epochs", type=int)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--output", default="outputs/few_shot_probes")
    args = p.parse_args(argv)
    from nlidnet.training.probes import run_probe

    for seed in args.seeds:
        for shots in args.shots:
            for head in args.heads:
                run_probe(
                    args.train_features,
                    args.test_features,
                    args.output,
                    "classification",
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


if __name__ == "__main__":
    main()
