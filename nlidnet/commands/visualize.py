import argparse
from pathlib import Path
import numpy as np
from nlidnet.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Temporal attention, event backgrounds, Grad-CAM caches and feature export"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    attention = commands.add_parser(
        "attention", help="Inspect temporal selection for a test trial"
    )
    attention.add_argument("--trial-id", required=True)
    features = commands.add_parser(
        "features", help="Export independent trial embeddings"
    )
    features.add_argument("--split", choices=["train", "test"], default="test")
    temporal = commands.add_parser(
        "export-attention",
        help="Export signals, logits, selected bins and attention for all trials",
    )
    temporal.add_argument("--split", choices=["train", "test"], default="test")
    gradcam = commands.add_parser(
        "gradcam",
        help="Cache activations/gradients and render overlays for selected trials",
    )
    gradcam.add_argument("--trial-ids", nargs="+", required=True)
    gradcam.add_argument(
        "--layers",
        nargs="+",
        default=[
            "pattern_encoder.backbone.layer1",
            "pattern_encoder.backbone.layer2",
            "pattern_encoder.backbone.layer3",
            "pattern_encoder.backbone.layer4",
        ],
    )
    gradcam.add_argument(
        "--class-index",
        type=int,
        help="Classifier output index; defaults to the predicted class",
    )
    gradcam.add_argument("--frame-indices", nargs="+", type=int)
    gradcam.add_argument("--alpha", type=float, default=0.55)
    gradcam.add_argument(
        "--background", choices=["unweighted", "weighted"], default="unweighted"
    )
    render = commands.add_parser(
        "render-gradcam", help="Render an existing cache without loading the model"
    )
    render.add_argument("--cache", required=True)
    render.add_argument("--layers", nargs="+")
    render.add_argument("--frame-indices", nargs="+", type=int)
    render.add_argument("--alpha", type=float, default=0.55)
    render.add_argument(
        "--background", choices=["unweighted", "weighted"], default="unweighted"
    )
    render.add_argument("--output", default="outputs/gradcam_render")
    video = commands.add_parser(
        "video", help="Encode timestamp-indexed frame PNGs as an H.264 MP4"
    )
    video.add_argument("--frames", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--fps", type=float, default=50)
    video.add_argument("--delta-t-us", type=int, default=2000)
    video.add_argument("--crop", nargs=4, type=int)
    video.add_argument("--accumulate", type=int, default=1)
    flow_video = commands.add_parser(
        "flow-video",
        help="Render paired event counts and flow fields using cache timestamps",
    )
    flow_video.add_argument("--cache-root", required=True)
    flow_video.add_argument("--output", required=True)
    flow_video.add_argument("--sample-step", type=int, default=10)
    flow_video.add_argument("--slowdown", type=float, default=5)
    flow_video.add_argument("--gif")
    gallery = commands.add_parser(
        "gallery",
        help="Render overview figures from temporal, Grad-CAM or feature caches",
    )
    gallery.add_argument(
        "--kind", choices=["temporal", "gradcam", "embeddings"], required=True
    )
    gallery.add_argument("--caches", nargs="+", required=True)
    gallery.add_argument("--output", required=True)
    gallery.add_argument("--method", choices=["pca", "tsne"], default="tsne")
    gallery.add_argument("--seed", type=int, default=42)
    gallery.add_argument("--frame-indices", nargs="+", type=int, default=[2, 7, 12, 17])
    gallery.add_argument("--segments", type=int, default=20)
    gallery.add_argument("--delta-t-us", type=int, default=2000)
    for command in (attention, features, temporal, gradcam):
        command.add_argument("--config", default="configs/paper.yaml")
        command.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
        command.add_argument("--device")
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "gallery":
        from nlidnet.visualization.gallery import (
            temporal_gallery,
            gradcam_gallery,
            embedding_gallery,
        )

        if args.kind == "temporal":
            return temporal_gallery(
                args.caches, args.output, args.segments, args.delta_t_us
            )
        if args.kind == "gradcam":
            return gradcam_gallery(args.caches, args.output, args.frame_indices)
        if len(args.caches) != 1:
            parser.error("Embedding gallery expects one feature file")
        return embedding_gallery(args.caches[0], args.output, args.method, args.seed)
    if args.command == "video":
        from nlidnet.visualization.video import frames_video

        return frames_video(
            args.frames,
            args.output,
            args.fps,
            args.delta_t_us,
            args.crop,
            args.accumulate,
        )
    if args.command == "flow-video":
        from nlidnet.visualization.video import flow_cache_video

        return flow_cache_video(
            args.cache_root, args.output, args.sample_step, args.slowdown, args.gif
        )
    if args.command == "render-gradcam":
        from nlidnet.visualization.gradcam import render_gradcam

        return render_gradcam(
            args.cache,
            args.output,
            args.layers,
            args.frame_indices,
            args.background,
            args.alpha,
        )
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    from nlidnet.visualization.export import (
        export_features,
        export_attention,
        inspect_trial,
    )

    if args.command == "attention":
        return inspect_trial(
            cfg, args.checkpoint, args.trial_id, args.output, "attention"
        )
    if args.command == "features":
        return export_features(cfg, args.checkpoint, args.output, args.split)
    if args.command == "export-attention":
        return export_attention(cfg, args.checkpoint, args.output, args.split)
    from nlidnet.data.dataset import LiquidEventDataset, collate_trials
    from nlidnet.protocol import LIQUIDS
    from nlidnet.utils import build_model, move_inputs, save_json
    from nlidnet.visualization.gradcam import capture_gradcam, render_gradcam

    model = build_model(cfg, args.checkpoint).to(cfg["device"]).eval()
    dataset = LiquidEventDataset(cfg, "test", tuple(range(len(LIQUIDS))), offset=0)
    by_id = {row["trial_id"]: i for i, row in enumerate(dataset.records)}
    missing = set(args.trial_ids) - set(by_id)
    if missing:
        parser.error(f"Unknown trial IDs: {sorted(missing)}")
    for trial_id in args.trial_ids:
        batch = collate_trials([dataset[by_id[trial_id]]])
        cache = capture_gradcam(
            model, move_inputs(batch, cfg["device"]), args.layers, args.class_index
        )
        out = Path(args.output) / f"trial_{by_id[trial_id]:06d}"
        out.mkdir(parents=True, exist_ok=True)
        cache_path = out / "gradcam.npz"
        if cache_path.exists():
            raise FileExistsError(cache_path)
        np.savez_compressed(cache_path, **cache)
        save_json(
            out / "trial.json",
            {
                "trial_id": trial_id,
                "raw_label": int(batch["raw_labels"][0]),
                "stage": cfg["stage"],
                "target_class_index": int(cache["class_index"]),
            },
        )
        render_gradcam(
            cache, out, args.layers, args.frame_indices, args.background, args.alpha
        )


if __name__ == "__main__":
    main()
