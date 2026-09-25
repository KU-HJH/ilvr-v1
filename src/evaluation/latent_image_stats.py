"""Question-clustered statistics for latent image analysis."""
from math import comb

import numpy as np


def cosine_distance(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1 - num / np.maximum(den, 1e-12)


_CHUNK = 2000


def _cluster_multiplicities(n_clusters, n, seed):
    """Vectorised replacement for drawing `n` cluster-bootstrap resamples.

    The original scheme drew, for each of `n` resamples, `n_clusters` cluster ids
    uniformly with replacement and concatenated their whole unit-index arrays. The
    *count* of times each cluster id is chosen in one such draw is distributionally
    identical to drawing `n_clusters` iid uniform ids and counting occurrences per
    id (though not bit-identical, since it does not walk the same RNG stream as the
    original per-draw `rng.choice`). We draw the ids with `rng.integers` (fast,
    uniform) and turn them into per-cluster counts with a single flattened
    `np.bincount`, which is far faster than `rng.multinomial` once n_clusters is
    more than a handful (multinomial's per-category sampling cost is O(n_clusters)
    per draw). Processed in chunks so memory stays bounded (chunk x n_clusters ints)
    even when n_clusters is large.
    """
    if n_clusters == 0:
        # Preserve the original behaviour: resampling from zero clusters raised
        # ValueError (np.concatenate([]) on an empty per-draw index list).
        raise ValueError("cannot cluster-bootstrap: no clusters to resample")
    rng = np.random.default_rng(seed)
    start = 0
    while start < n:
        take = min(_CHUNK, n - start)
        chosen = rng.integers(0, n_clusters, size=(take, n_clusters))
        row_offset = np.arange(take)[:, None] * n_clusters
        flat_counts = np.bincount((chosen + row_offset).ravel(), minlength=take * n_clusters)
        yield flat_counts.reshape(take, n_clusters)
        start += take


def _draws_from_sums(n_clusters, n, seed, numer_sums, denom_sums):
    """Resample draws = (mult @ numer_sums) / (mult @ denom_sums) for each of the n
    resamples, computed in chunks via _cluster_multiplicities."""
    draws = np.empty(n, dtype=np.float64)
    pos = 0
    for mult in _cluster_multiplicities(n_clusters, n, seed):
        k = mult.shape[0]
        draws[pos:pos + k] = (mult @ numer_sums) / (mult @ denom_sums)
        pos += k
    return draws


def _summary(point, draws, values, clusters):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi),
            "n_units": int(len(values)), "n_clusters": int(len(np.unique(clusters)))}


def cluster_bootstrap(values, clusters, n=10000, seed=42):
    values, clusters = np.asarray(values, np.float64), np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    ids, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = len(ids)
    sums = np.bincount(inverse, weights=values, minlength=n_clusters)
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    draws = _draws_from_sums(n_clusters, n, seed, sums, counts)
    return _summary(values.mean(), draws, values, clusters)


def paired_cluster_bootstrap(a, b, clusters, n=10000, seed=42):
    return cluster_bootstrap(np.asarray(a, np.float64) - np.asarray(b, np.float64), clusters, n, seed)


def ratio_cluster_bootstrap(num, den, clusters, n=10000, seed=42):
    num, den, clusters = np.asarray(num, np.float64), np.asarray(den, np.float64), np.asarray(clusters)
    keep = ~(np.isnan(num) | np.isnan(den))
    num, den, clusters = num[keep], den[keep], clusters[keep]
    ids, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = len(ids)
    num_sums = np.bincount(inverse, weights=num, minlength=n_clusters)
    den_sums = np.bincount(inverse, weights=den, minlength=n_clusters)
    draws = _draws_from_sums(n_clusters, n, seed, num_sums, den_sums)
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
