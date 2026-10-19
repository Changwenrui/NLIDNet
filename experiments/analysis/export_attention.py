import argparse
from experiments._arguments import add_config, configuration


def main():
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--output", default="outputs/temporal_attention")
    args = p.parse_args()
    from nlidnet.visualization.export import export_attention

    export_attention(configuration(args), args.checkpoint, args.output, args.split)


if __name__ == "__main__":
    main()
