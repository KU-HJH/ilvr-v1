"""Question-clustered statistics for latent image analysis."""
from math import comb

import numpy as np


def cosine_distance(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1 - num / np.maximum(den, 1e-12)


def _resample(clusters, n, seed):
    ids = np.unique(clusters)
    rng = np.random.default_rng(seed)
    index = {c: np.flatnonzero(clusters == c) for c in ids}
    for _ in range(n):
        yield np.concatenate([index[c] for c in rng.choice(ids, size=len(ids), replace=True)])


def _summary(point, draws, values, clusters):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi),
            "n_units": int(len(values)), "n_clusters": int(len(np.unique(clusters)))}


def cluster_bootstrap(values, clusters, n=10000, seed=42):
    values, clusters = np.asarray(values, np.float64), np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    draws = [values[i].mean() for i in _resample(clusters, n, seed)]
    return _summary(values.mean(), draws, values, clusters)


def paired_cluster_bootstrap(a, b, clusters, n=10000, seed=42):
    return cluster_bootstrap(np.asarray(a, np.float64) - np.asarray(b, np.float64), clusters, n, seed)


def ratio_cluster_bootstrap(num, den, clusters, n=10000, seed=42):
    num, den, clusters = np.asarray(num, np.float64), np.asarray(den, np.float64), np.asarray(clusters)
    keep = ~(np.isnan(num) | np.isnan(den))
    num, den, clusters = num[keep], den[keep], clusters[keep]
    draws = [num[i].sum() / den[i].sum() for i in _resample(clusters, n, seed)]
    return _summary(num.sum() / den.sum(), draws, num, clusters)


def exact_mcnemar(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def identification_accuracy(latent, image):
    latent, image = np.asarray(latent, np.float64), np.asarray(image, np.float64)
    latent = latent - latent.mean(1, keepdims=True)
    image = image - image.mean(1, keepdims=True)
    latent /= np.maximum(np.linalg.norm(latent, axis=-1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=-1, keepdims=True), 1e-12)
    sims = np.einsum("qid,qjd->qij", latent, image)
    return float((sims.argmax(-1) == np.arange(latent.shape[1])[None]).mean())
