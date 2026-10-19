from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from nlidnet.data.dataset import LiquidEventDataset, collate_trials
from nlidnet.protocol import LIQUIDS
from nlidnet.utils import build_model, move_inputs, aggregate_trials
from nlidnet.visualization.plotting import plot_attention, plot_embeddings


def export_features(cfg, checkpoint, output, split="test"):
    model = build_model(cfg, checkpoint).to(cfg["device"]).eval()
    features = []
    labels = []
    ids = []
    with torch.no_grad():
        for offset in cfg["eval_offsets"]:
            ds = LiquidEventDataset(
                cfg, split, tuple(range(len(LIQUIDS))), offset=offset
            )
            for b in DataLoader(
                ds, batch_size=1, num_workers=cfg["workers"], collate_fn=collate_trials
            ):
                features.extend(model(*move_inputs(b, cfg["device"]))[2].cpu().numpy())
                labels.extend(b["raw_labels"].tolist())
                ids.extend(b["trial_ids"])
    features, labels, ids = aggregate_trials(np.asarray(features), labels, ids)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "embeddings.npz",
        features=features,
        labels=labels,
        trial_ids=np.asarray(ids),
    )
    plot_embeddings(features, labels, out / "embeddings_pca.png")


def inspect_trial(cfg, checkpoint, trial_id, output, kind="attention"):
    model = build_model(cfg, checkpoint).to(cfg["device"]).eval()
    ds = LiquidEventDataset(cfg, "test", tuple(range(len(LIQUIDS))), offset=0)
    matches = [i for i, r in enumerate(ds.records) if r["trial_id"] == trial_id]
    if not matches:
        raise ValueError(f"Trial not found in test split: {trial_id}")
    b = collate_trials([ds[matches[0]]])
    inputs = move_inputs(b, cfg["device"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if kind == "attention":
        with torch.no_grad():
            attention = model(*inputs)[1][0].cpu().numpy()
        signal = b["inputs"][0][0].numpy()
        plot_attention(signal, attention, out / "attention.png", cfg["delta_t_us"])
        np.savez_compressed(out / "attention.npz", signal=signal, attention=attention)
    else:
        from nlidnet.visualization.gradcam import gradcam
        import matplotlib.pyplot as plt

        maps, index = gradcam(model, inputs)
        np.savez_compressed(
            out / "gradcam.npz", maps=maps.cpu().numpy(), class_index=index
        )
        fig, axes = plt.subplots(
            1, len(maps), figsize=(max(8, len(maps)), 3), squeeze=False
        )
        for i, (ax, heatmap) in enumerate(zip(axes[0], maps.cpu().numpy())):
            ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
            ax.set_title(str(i + 1))
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "gradcam.png", dpi=180)
        plt.close(fig)


def export_attention(cfg, checkpoint, output, split="test"):
    import csv

    model = build_model(cfg, checkpoint).to(cfg["device"]).eval()
    data = LiquidEventDataset(cfg, split, tuple(range(len(LIQUIDS))), offset=0)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "label", "file"])
        with torch.no_grad():
            for index, batch in enumerate(
                DataLoader(
                    data,
                    batch_size=1,
                    num_workers=cfg["workers"],
                    collate_fn=collate_trials,
                )
            ):
                inputs = move_inputs(batch, cfg["device"])
                raw = model.temporal_encoder(inputs[0]).reshape(1, 1, -1)
                smooth = model.gaussian_smooth(raw)
                attention = torch.softmax(
                    smooth.reshape(1, cfg["T"], cfg["k"]), dim=-1
                ).reshape(-1)
                filename = f"{index:06d}.npz"
                np.savez_compressed(
                    out / filename,
                    signal=inputs[0][0].cpu().numpy(),
                    raw_logits=raw.reshape(-1).cpu().numpy(),
                    attention=attention.cpu().numpy(),
                )
                writer.writerow(
                    [batch["trial_ids"][0], int(batch["raw_labels"][0]), filename]
                )
