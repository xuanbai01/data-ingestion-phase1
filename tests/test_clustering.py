"""
Tests for the clustering quality layer in pipeline/feature_engineering.py.

These exercise pure functions only — no model loading. Synthetic embeddings
and aspect lists let us verify select_k picks the right K and
merge_similar_clusters collapses overlapping clusters.
"""
from collections import defaultdict

import numpy as np
import pytest

from pipeline.feature_engineering import (
    select_k,
    merge_similar_clusters,
    _overlap_coefficient,
    _aspect_doc_freq,
    _distinctive_aspects,
)


# ---------------------------------------------------------------------------
# select_k
# ---------------------------------------------------------------------------

def _well_separated_blobs(k, n_per_cluster=50, dim=8, sep=10.0, seed=0):
    """Synthesize K well-separated Gaussian blobs in `dim` dimensions."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(k, dim))
    points = []
    for c in centers:
        points.append(rng.normal(c, 1.0, size=(n_per_cluster, dim)))
    return np.vstack(points)


def test_select_k_recovers_true_k_on_clean_data():
    """
    On well-separated synthetic blobs the silhouette score peaks at the
    correct K. Use 5 blobs and verify select_k picks 5 (or close to it).
    """
    embeddings = _well_separated_blobs(k=5, n_per_cluster=80)
    chosen = select_k(embeddings, k_range=(4, 9), verbose=False)
    # Allow ±1 to absorb stochasticity in KMeans init / silhouette sampling
    assert 4 <= chosen <= 6, f"select_k returned {chosen}, expected ~5"


def test_select_k_returns_small_value_on_tiny_corpus():
    """Below the silhouette-reliability threshold (n<100), select_k returns a small fixed K."""
    embeddings = np.random.rand(20, 8)
    k = select_k(embeddings, verbose=False)
    assert 2 <= k <= 8


def test_select_k_caps_k_to_corpus_size():
    """K is bounded by n // 5 so each cluster has at least ~5 points on average."""
    embeddings = _well_separated_blobs(k=3, n_per_cluster=40)  # 120 points
    k = select_k(embeddings, k_range=(4, 50), verbose=False)
    assert k <= 120 // 5


# ---------------------------------------------------------------------------
# overlap coefficient + top aspect set
# ---------------------------------------------------------------------------

def test_overlap_coefficient_basic():
    a = {"checkout", "payment", "card", "error"}
    b = {"checkout", "payment", "phone", "screen"}
    # |A ∩ B| = 2, min(|A|, |B|) = 4 → 0.5
    assert _overlap_coefficient(a, b) == 0.5


def test_overlap_coefficient_empty_sets_safe():
    assert _overlap_coefficient(set(), {"a"}) == 0.0
    assert _overlap_coefficient({"a"}, set()) == 0.0
    assert _overlap_coefficient(set(), set()) == 0.0


def test_distinctive_aspects_filters_corpus_common_words():
    """
    Words appearing in every cluster have low IDF and should NOT be
    distinctive. Words concentrated in one cluster have high IDF and SHOULD.
    Regression test for the over-eager-merge bug seen on real Amazon data.
    """
    # 5 clusters of 40 reviews each. 'item' and 'service' appear everywhere.
    # 'tablet' is only in cluster A.
    reviews = []
    cluster_a = [{"aspects": ["item", "service", "tablet", "screen", "device"]} for _ in range(40)]
    cluster_b = [{"aspects": ["item", "service", "delivery", "package", "driver"]} for _ in range(40)]
    cluster_c = [{"aspects": ["item", "service", "account", "password", "login"]} for _ in range(40)]
    cluster_d = [{"aspects": ["item", "service", "search", "filter", "sort"]} for _ in range(40)]
    cluster_e = [{"aspects": ["item", "service", "checkout", "card", "payment"]} for _ in range(40)]
    reviews = cluster_a + cluster_b + cluster_c + cluster_d + cluster_e

    corpus_df = _aspect_doc_freq(reviews)
    distinctive = _distinctive_aspects(cluster_a, corpus_df, len(reviews), k=4)

    # 'tablet' is uniquely cluster A's → high distinctiveness
    assert "tablet" in distinctive
    # 'item' is corpus-common → not distinctive
    assert "item" not in distinctive
    # 'service' is corpus-common → not distinctive
    assert "service" not in distinctive


def test_distinctive_aspects_falls_back_for_small_clusters():
    """Below the IDF-stability threshold we use raw frequency."""
    small_cluster = [{"aspects": ["a", "b"]}, {"aspects": ["a", "c"]}]  # n=2
    corpus_df = _aspect_doc_freq(small_cluster)
    out = _distinctive_aspects(small_cluster, corpus_df, total_reviews=2, k=2)
    # Just frequency-based: 'a' is most common, then either 'b' or 'c'
    assert "a" in out


def test_merge_does_not_collapse_clusters_sharing_only_filler_vocab():
    """
    Two clusters with distinct cores (tablet vs delivery) but both mention
    common e-commerce filler ('item', 'service'). They should NOT merge —
    this is the regression for the K=12 → K=6 over-merge we saw on Amazon.
    """
    reviews = (
        [_review(0, ["tablet", "device", "screen", "android", "item", "service"]) for _ in range(40)]
        + [_review(1, ["delivery", "package", "driver", "late", "item", "service"]) for _ in range(40)]
        # Filler clusters establish 'item' and 'service' as corpus-common
        + [_review(2, ["item", "service", "search", "filter", "sort"]) for _ in range(40)]
        + [_review(3, ["item", "service", "account", "password", "login"]) for _ in range(40)]
        + [_review(4, ["item", "service", "checkout", "card", "payment"]) for _ in range(40)]
    )

    merge_similar_clusters(reviews, threshold=0.5, top_k=4, verbose=False)

    # Walk the surviving clusters and confirm tablet and delivery still
    # live in separate buckets — the bug was that they'd get merged via
    # the shared 'item'/'service' filler.
    cluster_to_aspects = defaultdict(set)
    for r in reviews:
        cluster_to_aspects[r["theme_cluster"]].update(r["aspects"])

    has_distinct_tablet = any(
        "tablet" in s and "delivery" not in s
        for s in cluster_to_aspects.values()
    )
    has_distinct_delivery = any(
        "delivery" in s and "tablet" not in s
        for s in cluster_to_aspects.values()
    )
    assert has_distinct_tablet, "tablet cluster got merged into something it shouldn't have"
    assert has_distinct_delivery, "delivery cluster got merged into something it shouldn't have"


# ---------------------------------------------------------------------------
# merge_similar_clusters
# ---------------------------------------------------------------------------

def _review(cluster, aspects):
    return {"theme_cluster": cluster, "aspects": list(aspects)}


def test_merge_collapses_overlapping_clusters():
    """
    Clusters 0 and 1 share 4 of 5 top aspects → overlap coef = 0.8 → merge.
    Cluster 2 is disjoint → stays.
    """
    reviews = (
        [_review(0, ["checkout", "card", "payment", "error", "account"]) for _ in range(20)]
        + [_review(1, ["checkout", "card", "payment", "error", "money"]) for _ in range(20)]
        + [_review(2, ["delivery", "package", "driver", "late"]) for _ in range(20)]
    )

    merge_similar_clusters(reviews, threshold=0.5, top_k=5, verbose=False)

    cluster_ids = {r["theme_cluster"] for r in reviews}
    assert len(cluster_ids) == 2  # 0+1 merged, 2 separate
    # IDs are renumbered contiguously starting at 0
    assert cluster_ids == {0, 1}


def test_merge_no_op_when_no_overlap():
    """All clusters thematically disjoint → no merges, IDs preserved (after renumbering)."""
    reviews = (
        [_review(0, ["checkout", "payment"]) for _ in range(10)]
        + [_review(1, ["delivery", "package"]) for _ in range(10)]
        + [_review(2, ["search", "filter"]) for _ in range(10)]
    )

    merge_similar_clusters(reviews, threshold=0.5, top_k=4, verbose=False)
    cluster_ids = {r["theme_cluster"] for r in reviews}
    assert cluster_ids == {0, 1, 2}


def test_merge_handles_transitive_pairs():
    """
    A~B and B~C but A and C don't directly overlap enough.
    Single-pass union-find still merges all three because each pair triggers union.
    """
    reviews = (
        [_review(0, ["a", "b", "c", "d"]) for _ in range(10)]      # shares a,b,c with cluster 1
        + [_review(1, ["a", "b", "c", "x"]) for _ in range(10)]    # shares a,b,c with both 0 and 2
        + [_review(2, ["a", "b", "x", "y"]) for _ in range(10)]    # shares a,b with cluster 1
    )

    merge_similar_clusters(reviews, threshold=0.5, top_k=4, verbose=False)
    cluster_ids = {r["theme_cluster"] for r in reviews}
    assert len(cluster_ids) == 1  # all collapsed via transitive merge


def test_merge_preserves_review_count():
    """Merging changes labels, never count."""
    reviews = (
        [_review(0, ["checkout", "card"]) for _ in range(15)]
        + [_review(1, ["checkout", "card"]) for _ in range(25)]
    )
    n_before = len(reviews)
    merge_similar_clusters(reviews, threshold=0.5, top_k=4, verbose=False)
    assert len(reviews) == n_before


def test_merge_renumbers_to_contiguous_ids():
    """After merge, surviving cluster IDs are 0..N-1 with no gaps."""
    reviews = (
        [_review(0, ["a", "b", "c"]) for _ in range(10)]
        + [_review(5, ["a", "b", "c"]) for _ in range(10)]    # merges with 0
        + [_review(10, ["x", "y", "z"]) for _ in range(10)]   # disjoint
    )
    merge_similar_clusters(reviews, threshold=0.5, top_k=3, verbose=False)
    cluster_ids = sorted({r["theme_cluster"] for r in reviews})
    assert cluster_ids == [0, 1]


def test_merge_skips_when_only_one_cluster():
    reviews = [_review(0, ["a", "b"]) for _ in range(10)]
    merge_similar_clusters(reviews, threshold=0.5, top_k=4, verbose=False)
    assert all(r["theme_cluster"] == 0 for r in reviews)


def test_merge_handles_none_clusters():
    """Reviews with theme_cluster=None are left untouched."""
    reviews = (
        [_review(0, ["a", "b"]) for _ in range(10)]
        + [_review(1, ["a", "b"]) for _ in range(10)]
        + [_review(None, ["c", "d"]) for _ in range(5)]
    )
    merge_similar_clusters(reviews, threshold=0.5, top_k=4, verbose=False)
    none_count = sum(1 for r in reviews if r["theme_cluster"] is None)
    assert none_count == 5
