from pathlib import Path
import numpy as np


def plot_attention(signal, attention, path, delta_t_us=2000):
    import matplotlib.pyplot as plt

    t = np.arange(len(attention)) * delta_t_us / 1e6
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True, layout="constrained")
    for i, row in enumerate(signal):
        axes[0].plot(t, row, lw=0.7, label=f"ROI {i+1}")
    axes[0].set_ylabel("Event count")
    axes[0].legend(ncol=6, frameon=False, fontsize=8)
    axes[1].plot(t, attention, color="#245789", lw=1)
    axes[1].set(xlabel="Time within stage window (s)", ylabel="Local attention")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_embeddings(features, labels, path, method="pca", seed=42):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if len(features) < 2:
        raise ValueError("At least two trials are required")
    if method == "pca":
        xy = PCA(n_components=2).fit_transform(features)
        axis_names = ("Principal component 1", "Principal component 2")
    elif method == "tsne":
        from sklearn.manifold import TSNE

        xy = TSNE(
            n_components=2,
            perplexity=min(30, max(1, (len(features) - 1) / 3)),
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(features)
        axis_names = ("t-SNE 1", "t-SNE 2")
    else:
        raise ValueError("method must be pca or tsne")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for label in sorted(set(labels)):
        keep = np.asarray(labels) == label
        ax.scatter(xy[keep, 0], xy[keep, 1], s=18, label=str(label), alpha=0.8)
    ax.set(xlabel=axis_names[0], ylabel=axis_names[1])
    ax.legend(title="Class", fontsize=7, frameon=False, ncol=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_temporal_selection(signal, selection, path, delta_t_us=2000):
    import matplotlib.pyplot as plt

    def array(tensor):
        return tensor.detach().cpu().numpy()[0]

    raw, smooth = array(selection.raw_logits), array(selection.smoothed_logits)
    attention = array(selection.fine_weights)
    coarse, keys = array(selection.segment_weights), array(
        selection.key_indices
    ).astype(int)
    time = np.arange(len(raw)) * delta_t_us / 1e6
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for i, row in enumerate(signal):
        axes[0].plot(time, row, linewidth=0.6, label=f"ROI {i + 1}")
    axes[0].legend(ncol=6, frameon=False, fontsize=8)
    axes[0].set_ylabel("Event count")
    axes[1].plot(time, raw, linewidth=0.6, color="#a4b2bc", label="Encoder logits")
    axes[1].plot(time, smooth, linewidth=1, color="#405b87", label="Smoothed logits")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Temporal score")
    axes[2].plot(time, attention, linewidth=0.8, color="#405b87")
    axes[2].scatter(time[keys], attention[keys], s=12, color="#a26539", zorder=3)
    axes[2].set_ylabel("Local weight")
    segment_width = len(raw) / len(coarse) * delta_t_us / 1e6
    centers = (np.arange(len(coarse)) + 0.5) * segment_width
    axes[3].bar(centers, coarse, width=segment_width * 0.85, color="#405b87")
    axes[3].set_ylabel("Segment weight")
    axes[3].set_xlabel("Time within stage window (s)")
    for ax in axes:
        for boundary in np.arange(1, len(coarse)) * segment_width:
            ax.axvline(boundary, linewidth=0.3, color="#c9ced1", zorder=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
