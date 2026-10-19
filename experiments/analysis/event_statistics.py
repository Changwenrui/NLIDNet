import argparse
from experiments._arguments import add_config, configuration


def main():
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--shots", type=int, choices=[1, 5, 10, 20])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="outputs/event_statistics.json")
    args = p.parse_args()
    from nlidnet.analysis.event_statistics import classify

    classify(configuration(args), args.output, args.shots, args.seed)


if __name__ == "__main__":
    main()
