import json
from pathlib import Path
import numpy as np


def temporal_gallery(caches, output, segments=20, delta_t_us=2000):
    import matplotlib.pyplot as plt

    if not caches or min(segments, delta_t_us) < 1:
        raise ValueError("Temporal gallery requires caches and positive bin settings")
    fig, axes = plt.subplots(
        2, len(caches), figsize=(4.2 * len(caches), 4.7), squeeze=False
    )
    for column, path in enumerate(caches):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            raw = data["raw_logits"].reshape(-1)
            smooth = data[
                "smooth_logits" if "smooth_logits" in data else "smoothed_logits"
            ].reshape(-1)
            attention = data[
                "timeattention" if "timeattention" in data else "attention"
            ].reshape(-1)
        if len(raw) != len(smooth) or len(raw) != len(attention) or len(raw) % segments:
            raise ValueError(f"Incompatible temporal cache: {path}")
        bins = len(raw) // segments
        grouped = smooth.reshape(segments, bins)
        scores = np.exp(grouped - grouped.max(axis=1, keepdims=True))
        expected = (scores / scores.sum(axis=1, keepdims=True)).reshape(-1)
        if not np.allclose(expected, attention, atol=2e-6):
            raise ValueError(
                f"Attention does not match smoothed encoder logits: {path}"
            )
        selected = np.arange(segments) * bins + grouped.argmax(axis=1)
        seconds = np.arange(len(raw)) * delta_t_us / 1e6
        axes[0, column].plot(
            seconds, raw, color="#b2bec8", linewidth=0.7, label="Encoder score"
        )
        axes[0, column].plot(
            seconds, smooth, color="#385b80", linewidth=1.1, label="Smoothed score"
        )
        axes[1, column].plot(seconds, attention, color="#385b80", linewidth=0.9)
        axes[1, column].scatter(
            seconds[selected], attention[selected], color="#ad693d", s=9, zorder=3
        )
        for boundary in np.arange(1, segments) * bins * delta_t_us / 1e6:
            axes[1, column].axvline(boundary, color="#cbd1d6", linewidth=0.4, zorder=0)
        meta_path = path.parent / "meta.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
        title = meta.get("stage_name", meta.get("stage", f"Stage {column + 1}"))
        axes[0, column].set_title(
            title.split("_")[0].replace("stage", "Stage "), fontsize=11
        )
        axes[1, column].set_xlabel("Time within observation window (s)")
        axes[1, column].set_ylim(bottom=0)
        for row in range(2):
            axes[row, column].spines[["top", "right"]].set_visible(False)
            axes[row, column].tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Temporal score")
    axes[1, 0].set_ylabel("Local attention")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def gradcam_gallery(caches, output, frame_indices=(2, 7, 12, 17), frame_ms=70):
    import matplotlib.pyplot as plt
    import torch
    from torch.nn import functional as F

    if not caches or not frame_indices:
        raise ValueError("Grad-CAM gallery requires caches and frame indices")
    fig, axes = plt.subplots(
        len(caches),
        len(frame_indices),
        figsize=(2.5 * len(frame_indices), 3.6 * len(caches)),
        squeeze=False,
    )
    for row, path in enumerate(caches):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "layer1_act" in data:
                activation = data["layer1_act"]
                gradient = data["layer1_grad"]
                saved_cam = data["cam_raw"]
                events = np.asarray(data["event_bg"], dtype=np.float32)
            else:
                key = next(
                    (key for key in data.files if key.endswith("layer1/activations")),
                    None,
                )
                if key is None:
                    raise ValueError(f"No pattern layer1 activations in cache: {path}")
                prefix = key.removesuffix("/activations")
                activation = data[key]
                gradient = data[prefix + "/gradients"]
                saved_cam = data[prefix + "/cam"]
                events = np.asarray(data["unweighted_events"], dtype=np.float32).sum(
                    axis=1
                )
            cam = np.maximum(
                (activation * gradient.mean(axis=(2, 3), keepdims=True)).sum(axis=1), 0
            )
            if not np.allclose(cam, saved_cam, atol=1e-7):
                raise ValueError(
                    f"Cached CAM does not match activations and gradients: {path}"
                )
        if any(index < 0 or index >= len(events) for index in frame_indices):
            raise ValueError("Frame index lies outside the cached sequence")
        maps = F.interpolate(
            torch.from_numpy(cam)[:, None],
            size=events.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        maps /= np.maximum(
            maps.reshape(len(maps), -1).max(axis=1)[:, None, None], 1e-12
        )
        positive = events[events > 0]
        scale = (
            max(float(np.percentile(np.log1p(positive), 99)), 1e-6)
            if len(positive)
            else 1
        )
        background = np.clip(np.log1p(events) / scale, 0, 1)
        meta_path = path.parent / "meta.json"
        if not meta_path.exists():
            meta_path = path.parent / "trial.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
        from nlidnet.protocol import LIQUIDS

        name = meta.get("class_name")
        if name is None and "raw_label" in meta:
            name = LIQUIDS[int(meta["raw_label"])]
        name = name or path.parent.name
        for column, index in enumerate(frame_indices):
            ax = axes[row, column]
            ax.imshow(background[index], cmap="gray", vmin=0, vmax=1)
            ax.imshow(
                maps[index], cmap="inferno", vmin=0, vmax=1, alpha=maps[index] * 0.65
            )
            ax.set_title(f"{(index + 0.5) * frame_ms / 1000:.3f} s", fontsize=10)
            if column == 0:
                ax.set_ylabel(name.replace("_", " "), fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def embedding_gallery(features_path, output, method="tsne", seed=42):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from nlidnet.analysis.features import load_features
    from nlidnet.protocol import LIQUIDS, NOVEL_LABELS

    data = load_features(features_path)
    if len(data.features) < 3:
        raise ValueError("At least three feature vectors are required")
    features = StandardScaler().fit_transform(data.features)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(1, (len(features) - 1) // 3))
        coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(features)
        axis_names = ("t-SNE 1", "t-SNE 2")
    elif method == "pca":
        coordinates = PCA(n_components=2, random_state=seed).fit_transform(features)
        axis_names = ("Principal component 1", "Principal component 2")
    else:
        raise ValueError("Choose tsne or pca")
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set2").colors)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    for index, label in enumerate(sorted(set(data.labels))):
        keep = data.labels == label
        name = (
            LIQUIDS[int(label)].replace("_", " ")
            if 0 <= label < len(LIQUIDS)
            else str(label)
        )
        marker = "^" if label in NOVEL_LABELS else "o"
        ax.scatter(
            coordinates[keep, 0],
            coordinates[keep, 1],
            s=25,
            color=colors[index % len(colors)],
            marker=marker,
            alpha=0.85,
            linewidths=0,
            label=name,
        )
    ax.set(
        xlabel=axis_names[0], ylabel=axis_names[1], title="Liquid response embeddings"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.12)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=7.5, frameon=False
    )
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return coordinates
