import argparse
from nlidnet.analysis.features import load_features
from nlidnet.visualization.plotting import plot_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--method", choices=["pca", "tsne"], default="pca")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/embeddings.png")
    args = p.parse_args()
    data = load_features(args.features)
    plot_embeddings(data.features, data.labels, args.output, args.method, args.seed)


if __name__ == "__main__":
    main()
