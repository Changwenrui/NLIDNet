import argparse
from nlidnet.analysis.features import load_features
from nlidnet.visualization.plotting import plot_embeddings as draw_embeddings
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


def physics(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--event-features", nargs="+", required=True)
    p.add_argument("--rgb-features", nargs="+", required=True)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/physics_correlation")
    args = p.parse_args(argv)
    if min(args.permutations, args.bootstraps) < 0 or args.components < 1:
        p.error("Invalid analysis budget")
    from nlidnet.analysis.physics_correlation import analyze

    analyze(
        args.event_features,
        args.rgb_features,
        args.output,
        args.components,
        args.permutations,
        args.bootstraps,
        args.seed,
    )


def embeddings(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--reference-features")
    p.add_argument("--output", default="outputs/embedding_quality.json")
    args = p.parse_args(argv)
    from nlidnet.analysis.embeddings import analyze

    analyze(args.features, args.output, args.reference_features)


def speed(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--output", default="outputs/inference_speed")
    args = p.parse_args(argv)
    from nlidnet.analysis.benchmark import benchmark

    benchmark(
        configuration(args),
        args.checkpoint,
        args.output,
        args.batches,
        args.warmup,
        args.batch_size,
        args.precision,
    )


def statistics(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--shots", type=int, choices=[1, 5, 10, 20])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="outputs/event_statistics.json")
    args = p.parse_args(argv)
    from nlidnet.analysis.event_statistics import classify

    classify(configuration(args), args.output, args.shots, args.seed)


def flow(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--events", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument(
        "--selection", choices=["uniform", "segment_peak"], default="segment_peak"
    )
    p.add_argument("--output", default="outputs/optical_flow")
    p.add_argument(
        "--dense",
        action="store_true",
        help="Export paired event_count.npy and flow_uv.npy for video rendering",
    )
    p.add_argument("--all-stages", action="store_true")
    p.add_argument("--context-px", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args(argv)
    cfg = configuration(args)
    if args.all_stages and not args.dense:
        p.error("--all-stages requires --dense")
    if args.dense:
        from nlidnet.analysis.optical_flow import export_flow_cache
        from nlidnet.protocol import STAGES

        stages = STAGES if args.all_stages else {cfg["stage"]: cfg["start_frame"]}
        return export_flow_cache(
            args.events,
            args.checkpoint or cfg["flow_checkpoint"],
            args.output,
            stages,
            cfg["window_bins"],
            cfg["delta_t_us"],
            args.context_px,
            args.batch_size,
            cfg["device"],
        )
    from nlidnet.analysis.optical_flow import export_flow

    export_flow(
        args.events,
        args.checkpoint or cfg["flow_checkpoint"],
        args.output,
        cfg["start_frame"],
        cfg["start_frame"] + cfg["window_bins"],
        args.frames,
        args.selection,
        cfg["device"],
    )


def plot_embeddings(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--method", choices=["pca", "tsne"], default="pca")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/embeddings.png")
    args = p.parse_args(argv)
    data = load_features(args.features)
    draw_embeddings(data.features, data.labels, args.output, args.method, args.seed)


def export_attention(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--output", default="outputs/temporal_attention")
    args = p.parse_args(argv)
    from nlidnet.visualization.export import export_attention

    export_attention(configuration(args), args.checkpoint, args.output, args.split)


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "physics": physics,
        "embeddings": embeddings,
        "speed": speed,
        "statistics": statistics,
        "flow": flow,
        "plot-embeddings": plot_embeddings,
        "export-attention": export_attention,
    }
    if not args or args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=list(commands))
        parser.print_help()
        return
    if args[0] not in commands:
        raise SystemExit(
            f"Unknown command: {args[0]}; choose from {', '.join(commands)}"
        )
    return commands[args[0]](args[1:])


if __name__ == "__main__":
    main()
