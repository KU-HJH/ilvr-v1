import time

import numpy as np
import pytest
from evaluation import latent_image_stats as st


def test_cosine_distance():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(st.cosine_distance(a, a), [0, 0], atol=1e-12)
    np.testing.assert_allclose(st.cosine_distance(a, a[::-1]), [1, 1], atol=1e-12)


def test_cluster_bootstrap_respects_clusters():
    values = np.array([1.0, 1.0, 1.0, 3.0])
    clusters = np.array([0, 0, 0, 1])
    out = st.cluster_bootstrap(values, clusters, n=2000)
    assert out["n_clusters"] == 2 and out["lo"] <= out["mean"] <= out["hi"]
    assert out["mean"] == pytest.approx(1.5)
    same = st.cluster_bootstrap(np.ones(6), np.arange(6), n=500)
    assert same["lo"] == same["hi"] == 1


def test_paired_and_ratio():
    c = np.arange(5)
    d = st.paired_cluster_bootstrap(np.full(5, 2.0), np.ones(5), c, n=200)
    assert d["mean"] == pytest.approx(1)
    r = st.ratio_cluster_bootstrap(np.full(5, 1.0), np.full(5, 4.0), c, n=200)
    assert r["mean"] == pytest.approx(0.25)


def test_mcnemar():
    assert st.exact_mcnemar(0, 0) == 1.0
    assert st.exact_mcnemar(0, 6) == pytest.approx(0.03125)


def test_identification_accuracy():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(20, 5, 16))
    assert st.identification_accuracy(image + 0.01 * rng.normal(size=image.shape), image) == 1.0
    assert st.identification_accuracy(rng.normal(size=image.shape), image) < 0.6


def test_cluster_bootstrap_degenerate_cases():
    with pytest.raises(ValueError):
        st.cluster_bootstrap(np.array([]), np.array([]), n=10)
    out = st.cluster_bootstrap(np.array([1.0, 2.0, 3.0]), np.zeros(3), n=50)
    assert out["lo"] == out["hi"] == pytest.approx(2.0)


def _reference_cluster_bootstrap(values, clusters, n, seed):
    """Direct port of the pre-vectorisation implementation (Python loop over n
    resamples, each concatenating whole-cluster index arrays), kept local to this
    test so it stays a fixed reference independent of any future stats.py changes."""
    values, clusters = np.asarray(values, np.float64), np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    ids = np.unique(clusters)
    rng = np.random.default_rng(seed)
    index = {c: np.flatnonzero(clusters == c) for c in ids}
    draws = []
    for _ in range(n):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([index[c] for c in chosen])
        draws.append(values[idx].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def test_cluster_bootstrap_vectorised_matches_reference_loop():
    """Equivalence test (R29): unequal cluster sizes, comparing the vectorised
    st.cluster_bootstrap against the reference Python-loop implementation above."""
    rng = np.random.default_rng(7)
    n_clusters = 30
    sizes = rng.integers(1, 15, size=n_clusters)
    clusters = np.repeat(np.arange(n_clusters), sizes)
    values = rng.normal(size=clusters.shape)

    ref_mean, ref_lo, ref_hi = _reference_cluster_bootstrap(values, clusters, n=4000, seed=42)
    out = st.cluster_bootstrap(values, clusters, n=4000, seed=42)

    assert out["mean"] == pytest.approx(ref_mean)
    assert abs(out["lo"] - ref_lo) < 0.02
    assert abs(out["hi"] - ref_hi) < 0.02


def test_cluster_bootstrap_speed_smoke():
    """Speed smoke test (R29): 100 clusters x 400 units, n=10000 draws must take < 1s."""
    rng = np.random.default_rng(3)
    n_clusters = 100
    units_per_cluster = 400
    clusters = np.repeat(np.arange(n_clusters), units_per_cluster)
    values = rng.normal(size=clusters.shape)

    start = time.perf_counter()
    st.cluster_bootstrap(values, clusters, n=10000, seed=42)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"cluster_bootstrap took {elapsed:.2f}s for 100x400 units, n=10000"
