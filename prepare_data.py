import argparse
import json
from pathlib import Path
import numpy as np
from nlidnet.data.indexing import (
    index_raw,
    index_processed,
    write_manifest,
    manifest_summary,
)
from nlidnet.data.raw import read_events, crop_events
from nlidnet.data.preprocessing import preprocess
from nlidnet.data.representations import frame_sequence
from nlidnet.protocol import STAGES, read_manifest


def prepare_representations(args):
    rows = read_manifest(args.manifest)
    for stage in args.stages:
        output = Path(args.output) / stage
        manifest_path = output / "trials.csv"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        prepared = []
        for row in rows:
            source = (
                Path(args.event_root) / row["split"] / (row["prefix"] + ".safetensors")
            )
            events = read_events(source)
            if not len(events):
                raise ValueError(f"Empty recording: {source}")
            window = crop_events(
                events,
                STAGES[stage] * args.delta_t_us,
                args.window_bins * args.delta_t_us,
            )
            relative_path = Path(row["split"]) / (row["prefix"] + ".npy")
            dest = output / relative_path
            if dest.exists():
                raise FileExistsError(dest)
            representation = frame_sequence(
                window,
                frames=args.frames,
                representation=args.representation,
                roi=tuple(args.roi),
                partition=args.partition,
                shift_fraction=args.shift_fraction,
                sample_fraction=args.sample_fraction,
                voxel_bins=args.voxel_bins,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, representation, allow_pickle=False)
            prepared.append({**row, "representation_path": relative_path.as_posix()})
            print(stage, row["trial_id"], representation.shape, flush=True)
        write_manifest(prepared, manifest_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RAW recordings, trial indexing, six-ROI signals and event representations"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser(
        "index-raw",
        help="Index liquid folders using an explicit split or a new sorted split",
    )
    raw.add_argument("--raw-root", required=True)
    raw.add_argument("--output", default="data/trials.csv")
    group = raw.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split-file",
        help="CSV containing raw_path, split and optional trial_id/prefix",
    )
    group.add_argument(
        "--test-per-class",
        type=int,
        help="Create a NEW split from naturally sorted filenames",
    )
    raw.add_argument(
        "--folder-map", help="JSON mapping canonical liquid names to source folders"
    )
    existing = commands.add_parser(
        "index-processed",
        help="Index existing paired PT/safetensors/TXT files without renaming",
    )
    existing.add_argument("--event-root", required=True)
    existing.add_argument("--roi-root")
    existing.add_argument("--output", default="data/trials.csv")
    prepare = commands.add_parser(
        "prepare",
        help="Convert indexed recordings to paired six-ROI signals and event tensors",
    )
    prepare.add_argument("--manifest", default="data/trials.csv")
    prepare.add_argument("--raw-root", required=True)
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--delta-t-us", type=int, default=2000)
    prepare.add_argument("--bins", type=int, default=9000)
    prepare.add_argument(
        "--labels",
        type=int,
        nargs="+",
        help="Process selected labels, including added ethanol concentrations",
    )
    prepare.add_argument("--splits", nargs="+", choices=["train", "test"])
    representation = commands.add_parser(
        "represent", help="Export time surfaces, polarity frames or voxel grids"
    )
    representation.add_argument("--manifest", default="data/trials.csv")
    representation.add_argument("--event-root", default="data/processed")
    representation.add_argument(
        "--stages", nargs="+", choices=list(STAGES), default=list(STAGES)
    )
    representation.add_argument("--output", default="data/representations")
    representation.add_argument(
        "--representation",
        choices=["time_surface", "event_frame", "voxel_grid"],
        default="time_surface",
    )
    representation.add_argument(
        "--partition", choices=["count", "time"], default="count"
    )
    representation.add_argument("--frames", type=int, default=20)
    representation.add_argument("--voxel-bins", type=int, default=5)
    representation.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=[520, 0, 840, 600],
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    representation.add_argument("--shift-fraction", type=float, default=0)
    representation.add_argument("--sample-fraction", type=float, default=1)
    representation.add_argument("--delta-t-us", type=int, default=2000)
    representation.add_argument("--window-bins", type=int, default=700)
    preview = commands.add_parser(
        "preview", help="Plot six-ROI signals and export a polarity animation"
    )
    preview.add_argument("--events", required=True)
    preview.add_argument("--output", default="outputs/data_preview")
    preview.add_argument("--stage", choices=list(STAGES), default="stage1")
    preview.add_argument("--frame-bins", type=int, default=10)
    check = commands.add_parser(
        "validate", help="Validate manifest and processed trial pairing"
    )
    check.add_argument("--config", default="configs/paper.yaml")
    check.add_argument("--stage", choices=list(STAGES), default="stage1")
    args = parser.parse_args(argv)
    if args.command == "index-raw":
        folders = (
            json.loads(Path(args.folder_map).read_text(encoding="utf-8"))
            if args.folder_map
            else None
        )
        rows = index_raw(
            args.raw_root, args.output, args.split_file, args.test_per_class, folders
        )
        print(json.dumps(manifest_summary(rows), indent=2))
    elif args.command == "index-processed":
        rows = index_processed(args.event_root, args.output, args.roi_root)
        print(json.dumps(manifest_summary(rows), indent=2))
    elif args.command == "prepare":
        return preprocess(
            args.manifest,
            args.raw_root,
            args.output,
            args.delta_t_us,
            args.bins,
            args.labels,
            args.splits,
        )
    elif args.command == "represent":
        return prepare_representations(args)
    elif args.command == "preview":
        from nlidnet.visualization.events import preview_recording

        return preview_recording(
            read_events(args.events),
            args.output,
            STAGES[args.stage],
            frame_bins=args.frame_bins,
        )
    else:
        from nlidnet.config import load_config
        from nlidnet.data.validate import validate_data

        return validate_data(load_config(args.config, args.stage))


if __name__ == "__main__":
    main()
