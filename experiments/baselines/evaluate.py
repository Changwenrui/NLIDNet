import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from nlidnet.models.baselines import build_event_model, build_video_model
from nlidnet.models.baselines.data import BaselineDataset
from nlidnet.utils import metrics, save_json
from nlidnet.protocol import STAGES


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=[
            "alexnet_lstm",
            "mobilenetv3_chem",
            "spiking_resformer",
            "videomae_v2",
            "go_with_the_flow",
            "dont_look_twice",
        ],
        required=True,
    )
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/baseline_evaluation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    video = args.model in {"videomae_v2", "go_with_the_flow", "dont_look_twice"}
    ds = BaselineDataset(
        args.manifest, args.data_root, "test", args.model, STAGES[args.stage] * 0.002
    )
    model = (
        build_video_model(args.model, pretrained=False)
        if video
        else build_event_model(args.model)
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.to(args.device).eval()
    scores = []
    labels = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=args.workers):
            scores.extend(model(x.to(args.device)).cpu().numpy())
            labels.extend(y.tolist())
            if args.model == "spiking_resformer":
                from spikingjelly.activation_based import functional

                functional.reset_net(model)
    result = metrics(np.asarray(scores), np.asarray(labels))
    out = Path(args.output) / args.model / args.stage
    save_json(out / "metrics.json", result)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "label", "prediction"])
        w.writerows(
            zip(
                [r["trial_id"] for r in ds.rows],
                labels,
                np.asarray(scores).argmax(axis=1),
            )
        )
    print(result)


if __name__ == "__main__":
    main()
