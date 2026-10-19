import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event-features", nargs="+", required=True)
    p.add_argument("--rgb-features", nargs="+", required=True)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/physics_correlation")
    args = p.parse_args()
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


if __name__ == "__main__":
    main()
