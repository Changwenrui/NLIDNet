import argparse
from experiments._arguments import add_config, configuration


def main():
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--output", default="outputs/inference_speed")
    args = p.parse_args()
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


if __name__ == "__main__":
    main()
