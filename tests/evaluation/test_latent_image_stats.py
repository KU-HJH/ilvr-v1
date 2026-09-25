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
