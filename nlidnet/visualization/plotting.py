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
