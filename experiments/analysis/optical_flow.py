import argparse
from experiments._arguments import add_config, configuration


def main():
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--events", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument(
        "--selection", choices=["uniform", "segment_peak"], default="segment_peak"
    )
    p.add_argument("--output", default="outputs/optical_flow")
    args = p.parse_args()
    cfg = configuration(args)
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


if __name__ == "__main__":
    main()
