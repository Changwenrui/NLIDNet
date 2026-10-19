import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from nlidnet.analysis.features import load_features
from nlidnet.utils import save_json
import argparse


def l2_normalize(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def compute_centroids(feats, labels):
    centroids = {}
    for y in sorted(set(labels.tolist())):
        centroids[int(y)] = feats[labels == y].mean(axis=0)
    return centroids


def nearest_centroid_accuracy(feats, labels, metric="cosine"):
    centroids = compute_centroids(feats, labels)
    cls = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in cls], axis=0)
    if metric == "cosine":
        D = cosine_distances(feats, C)
    else:
        D = euclidean_distances(feats, C)
    pred = np.asarray([cls[i] for i in D.argmin(axis=1)], dtype=np.int64)
    acc = float((pred == labels).mean())
    per_class = {}
    for y in sorted(set(labels.tolist())):
        idx = labels == y
        per_class[int(y)] = float((pred[idx] == labels[idx]).mean())
    return (acc, per_class)


def intra_inter_stats(feats, labels, metric="cosine"):
    centroids = compute_centroids(feats, labels)
    cls = sorted(centroids.keys())
    intra = {}
    for y in cls:
        x = feats[labels == y]
        c = centroids[y][None, :]
        if metric == "cosine":
            d = cosine_distances(x, c).reshape(-1)
        else:
            d = euclidean_distances(x, c).reshape(-1)
        intra[y] = float(d.mean())
    C = np.stack([centroids[c] for c in cls], axis=0)
    if metric == "cosine":
        D = cosine_distances(C, C)
    else:
        D = euclidean_distances(C, C)
    mask = ~np.eye(len(cls), dtype=bool)
    inter_mean = float(D[mask].mean())
    inter_min = float(D[mask].min())
    return (intra, inter_mean, inter_min)


def analyze(path, output, reference=None):
    data = load_features(path)
    features = l2_normalize(data.features)
    labels = data.labels
    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required")
    intra, inter_mean, inter_min = intra_inter_stats(features, labels)
    fitted, per_class = nearest_centroid_accuracy(features, labels)
    result = {
        "samples": len(labels),
        "classes": len(set(labels)),
        "dimension": features.shape[1],
        "centroid_fit_accuracy": fitted,
        "centroid_fit_per_class": per_class,
        "intra_distance": intra,
        "inter_centroid_mean": inter_mean,
        "inter_centroid_min": inter_min,
    }
    if 1 < len(set(labels)) < len(labels):
        result["silhouette_cosine"] = float(
            silhouette_score(features, labels, metric="cosine")
        )
    if reference:
        train = load_features(reference)
        centroids = compute_centroids(l2_normalize(train.features), train.labels)
        classes = sorted(centroids)
        matrix = np.stack([centroids[c] for c in classes])
        pred = np.asarray(classes)[cosine_distances(features, matrix).argmin(1)]
        known = np.isin(labels, classes)
        if known.any():
            result["reference_centroid_accuracy"] = float(
                (pred[known] == labels[known]).mean()
            )
        query_centroids = compute_centroids(features, labels)
        distances = cosine_distances(np.stack(list(query_centroids.values())), matrix)
        result["nearest_reference_classes"] = {
            int(label): [int(classes[j]) for j in np.argsort(distances[i])[:5]]
            for i, label in enumerate(query_centroids)
        }
        result["reference_classes"] = classes
    save_json(output, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--reference-features")
    p.add_argument("--output", default="outputs/embedding_quality.json")
    args = p.parse_args(argv)
    analyze(args.features, args.output, args.reference_features)


if __name__ == "__main__":
    main()
