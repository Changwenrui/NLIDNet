from pathlib import Path
from typing import Any, Dict, Tuple
import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nlidnet.analysis.features import combine_features
from nlidnet.data.physics import PHYS_RAW
from nlidnet.utils import save_json
import argparse

PROPERTY_NAMES = ["Density", "Viscosity", "Surface tension"]
PHYSICS_LABELS = tuple(range(12)) + tuple(range(21, 30))


def canonical_corr(U: np.ndarray, V: np.ndarray) -> float:
    r = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
    return 0.0 if np.isnan(r) else float(abs(r))


def fit_cca(
    X: np.ndarray, Y: np.ndarray, n_pc: int = 3, n_cca: int = 1, random_state: int = 42
) -> Dict[str, Any]:
    n = X.shape[0]
    max_pc = min(n_pc, X.shape[1], n - 1)
    if max_pc < 1:
        raise RuntimeError(f"n_pc too small after constraints: {max_pc}")
    sx, sy = (StandardScaler(), StandardScaler())
    Xz = sx.fit_transform(X)
    Yz = sy.fit_transform(Y)
    pca = PCA(n_components=max_pc, random_state=random_state)
    Xp = pca.fit_transform(Xz)
    n_comp = min(n_cca, max_pc, Yz.shape[1], n - 1)
    cca = CCA(n_components=n_comp, max_iter=5000, tol=1e-07)
    U, V = cca.fit_transform(Xp, Yz)
    r = canonical_corr(U, V)
    if np.corrcoef(U[:, 0], V[:, 0])[0, 1] < 0:
        V[:, 0] *= -1.0
    prop_loadings = []
    for j, prop in enumerate(PROPERTY_NAMES):
        rr = np.corrcoef(V[:, 0], Yz[:, j])[0, 1]
        prop_loadings.append(float(rr))
    prop_corr_u = []
    for j, prop in enumerate(PROPERTY_NAMES):
        pr, pp = pearsonr(U[:, 0], Y[:, j])
        sr, sp = spearmanr(U[:, 0], Y[:, j])
        prop_corr_u.append(
            {
                "property": prop,
                "pearson_r_with_U": float(pr),
                "pearson_p_with_U": float(pp),
                "spearman_r_with_U": float(sr),
                "spearman_p_with_U": float(sp),
            }
        )
    return {
        "r": r,
        "pca": pca,
        "cca": cca,
        "scaler_x": sx,
        "scaler_y": sy,
        "X_pca": Xp,
        "Y_scaled": Yz,
        "U": U,
        "V": V,
        "prop_loadings": np.asarray(prop_loadings, dtype=np.float64),
        "prop_corr_u": prop_corr_u,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "n_pc": max_pc,
        "n_cca": n_comp,
    }


def permutation_test_cca(
    X: np.ndarray, Y: np.ndarray, n_pc: int, n_perm: int, seed: int
) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    true = fit_cca(X, Y, n_pc=n_pc, random_state=seed)["r"]
    null = np.zeros(n_perm, dtype=np.float64)
    n = Y.shape[0]
    for i in range(n_perm):
        perm = rng.permutation(n)
        try:
            null[i] = fit_cca(X, Y[perm], n_pc=n_pc, random_state=seed)["r"]
        except Exception:
            null[i] = np.nan
    null = null[np.isfinite(null)]
    p = (1.0 + np.sum(null >= true)) / (1.0 + len(null))
    return (float(p), null)


def bootstrap_ci_cca(
    X: np.ndarray, Y: np.ndarray, n_pc: int, n_boot: int, seed: int
) -> Tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < max(5, n_pc + 2):
            continue
        try:
            vals.append(fit_cca(X[idx], Y[idx], n_pc=n_pc, random_state=seed)["r"])
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan, np.asarray([]))
    vals = np.asarray(vals, dtype=np.float64)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi), vals)


def permutation_test_delta(
    X_event: np.ndarray,
    X_rgb: np.ndarray,
    Y: np.ndarray,
    n_pc: int,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    r_e = fit_cca(X_event, Y, n_pc=n_pc, random_state=seed)["r"]
    r_r = fit_cca(X_rgb, Y, n_pc=n_pc, random_state=seed)["r"]
    delta_true = r_e - r_r
    n = Y.shape[0]
    null = np.zeros(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        Yp = Y[perm]
        try:
            re = fit_cca(X_event, Yp, n_pc=n_pc, random_state=seed)["r"]
            rr = fit_cca(X_rgb, Yp, n_pc=n_pc, random_state=seed)["r"]
            null[i] = re - rr
        except Exception:
            null[i] = np.nan
    null = null[np.isfinite(null)]
    if delta_true >= 0:
        p_one = (1.0 + np.sum(null >= delta_true)) / (1.0 + len(null))
    else:
        p_one = (1.0 + np.sum(null <= delta_true)) / (1.0 + len(null))
    p_two = (1.0 + np.sum(np.abs(null) >= abs(delta_true))) / (1.0 + len(null))
    return (float(p_one), float(p_two), null)


def analyze(event_paths, rgb_paths, output, n_pc=3, n_perm=5000, n_boot=2000, seed=42):
    event = combine_features(event_paths)
    rgb = combine_features(rgb_paths)
    labels = sorted(set(event.labels) & set(rgb.labels) & set(PHYSICS_LABELS))
    if len(labels) < max(5, n_pc + 2):
        raise ValueError("Insufficient shared liquid conditions for CCA")
    xe = np.stack([event.features[event.labels == label].mean(0) for label in labels])
    xr = np.stack([rgb.features[rgb.labels == label].mean(0) for label in labels])
    y = np.asarray([PHYS_RAW[label] for label in labels], dtype=float)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"conditions": len(labels), "n_pc": n_pc, "seed": seed}
    for modality, x in [("event", xe), ("rgb", xr)]:
        fitted = fit_cca(x, y, n_pc=n_pc, random_state=seed)
        row = {
            "canonical_correlation": fitted["r"],
            "property_loadings": fitted["prop_loadings"].tolist(),
        }
        if n_perm:
            p, null = permutation_test_cca(x, y, n_pc, n_perm, seed)
            row["permutation_p"] = p
            np.save(out / f"{modality}_permutation.npy", null)
        if n_boot:
            low, high, values = bootstrap_ci_cca(x, y, n_pc, n_boot, seed)
            row["bootstrap_ci"] = [low, high]
            np.save(out / f"{modality}_bootstrap.npy", values)
        summary[modality] = row
        with (out / f"{modality}_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["label", "feature_score", "property_score"])
            w.writerows(zip(labels, fitted["U"][:, 0], fitted["V"][:, 0]))
    if n_perm:
        one, two, null = permutation_test_delta(xe, xr, y, n_pc, n_perm, seed)
        summary["difference_test"] = {"one_sided_p": one, "two_sided_p": two}
        np.save(out / "difference_permutation.npy", null)
    save_json(out / "correlation.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--event-features", nargs="+", required=True)
    p.add_argument("--rgb-features", nargs="+", required=True)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/physics_correlation")
    args = p.parse_args(argv)
    if min(args.permutations, args.bootstraps) < 0 or args.components < 1:
        p.error("Invalid analysis budget")
    analyze(
        args.event_features,
        args.rgb_features,
        args.output,
        args.components,
        args.permutations,
        args.bootstraps,
        args.seed,
    )


if __name__ == "__main__":
    main()
