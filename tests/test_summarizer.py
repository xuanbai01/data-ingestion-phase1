"""
Tests for pipeline/summarizer.py.

Focused on the pure logic that's easy to test without loading models or
hitting an LLM: priority scoring, subjectivity split, distinctive aspects,
representative review selection, ABSA aggregation, snapshot building, and
the run-delta renderer. The renderers' top-level orchestration in
`generate_report` is exercised by the existing end-to-end smoke tests.
"""
import os

# Ensure the LLM module short-circuits during summarizer tests (no API key,
# fall back to aspect-string labels). Has to happen before importing the
# summarizer since pipeline/llm.py reads the env on import-of-anthropic.
os.environ.pop("ANTHROPIC_API_KEY", None)

import numpy as np
import pytest

from pipeline import summarizer as summ


# ---------------------------------------------------------------------------
# _aspect_names — backwards compatibility across the v2 → v3 schema bump
# ---------------------------------------------------------------------------

def test_aspect_names_handles_legacy_string_format():
    """Phase II reviews stored aspects as list[str]."""
    review = {"aspects": ["delivery", "package", "driver"]}
    assert summ._aspect_names(review) == ["delivery", "package", "driver"]


def test_aspect_names_handles_phase3_dict_format():
    """Phase III reviews store aspects as list[{aspect, polarity, confidence}]."""
    review = {"aspects": [
        {"aspect": "delivery", "polarity": -0.8, "confidence": 0.95},
        {"aspect": "package", "polarity": 0.3, "confidence": 0.7},
    ]}
    assert summ._aspect_names(review) == ["delivery", "package"]


def test_aspect_names_empty_when_missing():
    assert summ._aspect_names({}) == []
    assert summ._aspect_names({"aspects": None}) == []


# ---------------------------------------------------------------------------
# _aspect_doc_freq + _distinctive_aspects (summarizer's variant)
# ---------------------------------------------------------------------------

def test_aspect_doc_freq_counts_unique_per_review():
    """A review mentioning 'delivery' twice still counts once toward DF."""
    reviews = [
        {"aspects": ["delivery", "delivery", "package"]},
        {"aspects": ["delivery"]},
        {"aspects": ["package"]},
    ]
    df = summ._aspect_doc_freq(reviews)
    assert df["delivery"] == 2
    assert df["package"] == 2


def test_distinctive_aspects_falls_back_for_small_clusters():
    """Below TFIDF_MIN_CLUSTER_SIZE (20), fall back to raw frequency."""
    cluster = [{"aspects": ["a", "b"]}, {"aspects": ["a", "c"]}]
    df = summ._aspect_doc_freq(cluster)
    out = summ._distinctive_aspects(cluster, df, total_reviews=2, k=2)
    assert "a" in out


def test_distinctive_aspects_uses_idf_for_large_clusters():
    """Corpus-common aspects (low IDF) drop out for ≥20-review clusters."""
    cluster_a = [{"aspects": ["common", "tablet"]} for _ in range(25)]
    other = [{"aspects": ["common", "x"]} for _ in range(25)] + \
            [{"aspects": ["common", "y"]} for _ in range(25)]
    all_reviews = cluster_a + other
    df = summ._aspect_doc_freq(all_reviews)

    distinctive = summ._distinctive_aspects(cluster_a, df, total_reviews=len(all_reviews), k=4)
    assert "tablet" in distinctive  # cluster-specific → high IDF
    assert "common" not in distinctive  # corpus-wide → IDF=0, dropped


# ---------------------------------------------------------------------------
# _score_issues — priority math + qualification gate
# ---------------------------------------------------------------------------

def _r(cid, rating, polarity=-0.3, urgency=0.5, emotion="anger", aspects=("a", "b")):
    return {
        "theme_cluster": cid,
        "rating": rating,
        "polarity": polarity,
        "urgency": urgency,
        "emotion": emotion,
        "aspects": list(aspects),
    }


def test_score_issues_excludes_positive_clusters():
    """A cluster with high rating and positive polarity is not an issue."""
    reviews = [_r(0, rating=5, polarity=0.5) for _ in range(10)]
    issues = summ._score_issues(reviews)
    assert issues == []


def test_score_issues_includes_low_rating_cluster():
    """A cluster with avg rating < ISSUE_RATING_CEILING qualifies."""
    reviews = [_r(0, rating=1, polarity=-0.5) for _ in range(10)]
    issues = summ._score_issues(reviews)
    assert len(issues) == 1
    assert issues[0]["cluster_id"] == 0


def test_score_issues_includes_negative_polarity_cluster_even_if_rating_ok():
    """A high-rating but very-negative-polarity cluster still qualifies (sarcasm catch)."""
    reviews = [_r(0, rating=4, polarity=-0.5) for _ in range(10)]
    issues = summ._score_issues(reviews)
    assert len(issues) == 1


def test_score_issues_ranks_by_priority_score():
    """Higher-priority issue should sort first."""
    big_severe = [_r(0, rating=1, polarity=-0.6, urgency=0.9) for _ in range(50)]
    small_mild = [_r(1, rating=2, polarity=-0.2, urgency=0.3) for _ in range(10)]
    issues = summ._score_issues(big_severe + small_mild)
    assert issues[0]["cluster_id"] == 0
    assert issues[0]["score"] > issues[1]["score"]


def test_score_issues_attaches_component_breakdown():
    """Components dict exposed for downstream rendering / debugging."""
    reviews = [_r(0, rating=1, polarity=-0.5, urgency=0.7) for _ in range(20)]
    issues = summ._score_issues(reviews)
    components = issues[0]["components"]
    for key in ("volume", "severity", "urgency", "emotion"):
        assert key in components
        assert 0.0 <= components[key] <= 1.0


def test_score_issues_emotion_share_uses_intense_only():
    """emotion_intensity is the share of intense emotions (anger/disgust/fear), not total."""
    intense = [_r(0, rating=1, emotion="anger") for _ in range(10)]
    mild = [_r(0, rating=1, emotion="sadness") for _ in range(10)]
    issues = summ._score_issues(intense + mild)
    # 10 of 20 are intense → 0.5
    assert abs(issues[0]["emotion_intensity"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _subjectivity_split
# ---------------------------------------------------------------------------

def test_subjectivity_split_partitions_at_threshold():
    """Reviews below SUBJECTIVE_THRESHOLD go to objective; above to subjective."""
    cluster = [
        {"subjectivity": 0.1},   # objective
        {"subjectivity": 0.49},  # objective (just under 0.5)
        {"subjectivity": 0.5},   # subjective (boundary inclusive on the 'subjective' side)
        {"subjectivity": 0.9},   # subjective
    ]
    obj, subj = summ._subjectivity_split(cluster)
    assert len(obj) == 2
    assert len(subj) == 2


def test_subjectivity_split_drops_none_values():
    """A review with subjectivity=None is excluded from both buckets."""
    cluster = [{"subjectivity": None}, {"subjectivity": 0.2}, {"subjectivity": 0.8}]
    obj, subj = summ._subjectivity_split(cluster)
    assert len(obj) == 1
    assert len(subj) == 1


# ---------------------------------------------------------------------------
# _representative_reviews — A2 thumbs-up-weighted selection
# ---------------------------------------------------------------------------

def _embedded(body, axis0, thumbs=0):
    """Synthesize a review with a 4-D embedding whose first axis encodes 'distance'."""
    emb = np.zeros(4, dtype=np.float32)
    emb[0] = axis0
    return {"body": body, "embedding": emb.tolist(), "thumbs_up": thumbs}


def test_representative_reviews_returns_empty_when_no_embeddings():
    cluster = [{"body": "x", "embedding": None}]
    assert summ._representative_reviews(cluster, n=3) == []


def test_representative_reviews_falls_back_to_closeness_when_zero_thumbs():
    """All thumbs=0 → pure-closeness order, matches pre-A2 behavior."""
    np.random.seed(0)
    embeds = np.random.randn(20, 4) * 0.5
    cluster = [
        {"body": f"r_{i}", "embedding": embeds[i].tolist(), "thumbs_up": 0}
        for i in range(20)
    ]
    centroid = embeds.mean(axis=0)
    distances = np.linalg.norm(embeds - centroid, axis=1)
    expected = [f"r_{i}" for i in np.argsort(distances)[:3]]

    picks = summ._representative_reviews(cluster, n=3)
    assert picks == expected


def test_representative_reviews_promotes_high_thumbs_within_pool():
    """A high-thumbs review inside the closeness pool should rise to the top."""
    np.random.seed(0)
    embeds = np.random.randn(20, 4) * 0.5
    cluster = [
        {"body": f"r_{i}", "embedding": embeds[i].tolist(), "thumbs_up": 0}
        for i in range(20)
    ]
    centroid = embeds.mean(axis=0)
    distances = np.linalg.norm(embeds - centroid, axis=1)
    # 5th-closest = inside the top quarter (pool_size = 20 // 4 = 5)
    boost_idx = np.argsort(distances)[4]
    cluster[boost_idx]["thumbs_up"] = 1000
    cluster[boost_idx]["body"] = "boosted"

    picks = summ._representative_reviews(cluster, n=3)
    assert "boosted" in picks


def test_representative_reviews_excludes_far_outlier_even_with_thumbs():
    """Candidate pool = closeness top quarter, so a far outlier never enters."""
    np.random.seed(0)
    embeds = np.random.randn(20, 4) * 0.5
    cluster = [
        {"body": f"r_{i}", "embedding": embeds[i].tolist(), "thumbs_up": 0}
        for i in range(20)
    ]
    centroid = embeds.mean(axis=0)
    distances = np.linalg.norm(embeds - centroid, axis=1)
    # The most-distant review gets a huge thumbs count — must still be excluded.
    far_idx = np.argsort(distances)[-1]
    cluster[far_idx]["thumbs_up"] = 5000
    cluster[far_idx]["body"] = "far_outlier"

    picks = summ._representative_reviews(cluster, n=3)
    assert "far_outlier" not in picks


def test_representative_reviews_dedupes_identical_bodies():
    """Output never contains the same body twice, even when duplicates exist
    in the candidate pool. (May return fewer than n if the pool collapses
    after dedup — that's correct behavior for tiny clusters.)
    """
    cluster = [
        _embedded("same", axis0=0.0),
        _embedded("same", axis0=0.05),  # duplicate body, slightly different position
        _embedded("alpha", axis0=0.1),
        _embedded("beta", axis0=0.2),
        _embedded("gamma", axis0=0.3),
        _embedded("delta", axis0=0.4),
        _embedded("epsilon", axis0=0.5),
        _embedded("zeta", axis0=0.6),
    ]
    picks = summ._representative_reviews(cluster, n=3)
    assert len(set(picks)) == len(picks), "no duplicate bodies"


def test_representative_reviews_truncates_long_bodies():
    """Long review bodies are capped at max_len (ellipsis included in budget)."""
    cluster = [
        {"body": "X" * 1000, "embedding": [0.0] * 4, "thumbs_up": 0},
        {"body": "Y" * 1000, "embedding": [0.1] * 4, "thumbs_up": 0},
        {"body": "Z" * 1000, "embedding": [0.2] * 4, "thumbs_up": 0},
    ]
    picks = summ._representative_reviews(cluster, n=3, max_len=50)
    for p in picks:
        assert len(p) <= 50


def test_truncate_at_word_breaks_at_space():
    """Phase X: truncation prefers word boundaries over mid-word cuts."""
    text = "the quick brown fox jumps over the lazy dog"
    out = summ._truncate_at_word(text, 25)
    assert out.endswith("…")
    # Last char before ellipsis should be a letter (end of a word), not mid-word
    assert out[-2].isalpha()
    # No trailing partial word like "ju..."
    assert "jumps over" in text  # sanity
    assert " " not in out[-3:-1]  # the chars right before "…" shouldn't include a stray space


def test_truncate_at_word_falls_back_for_unbreakable_input():
    """Single-word inputs longer than max_len fall back to hard cap."""
    text = "X" * 100
    out = summ._truncate_at_word(text, 30)
    assert len(out) <= 30
    assert out.endswith("…")


def test_truncate_at_word_passthrough_when_short():
    """Strings shorter than max_len pass through unchanged."""
    assert summ._truncate_at_word("hello", 100) == "hello"
    assert summ._truncate_at_word("", 100) == ""


# ---------------------------------------------------------------------------
# Phase X — entity noise filter
# ---------------------------------------------------------------------------

def test_render_entity_noise_drops_known_false_positives():
    """Spurious NER tags (Rufus, Customer Service, Newest, Tablet) get
    filtered at render time so they don't show up as 'mentioned entities'."""
    np.random.seed(0)
    e = (np.array([1, 0, 0, 0], dtype=np.float32) + np.random.randn(4) * 0.05).tolist()
    reviews = [
        {
            "reviewer_name": "x",
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "body": "issue review",
            "rating": 1, "polarity": -0.4, "subjectivity": 0.3,
            "urgency": 0.6, "emotion": "anger",
            "aspects": [{"aspect": "delivery", "polarity": -0.5, "confidence": 0.8}],
            "entities": [
                {"text": "Walmart", "label": "ORG"},          # legit
                {"text": "Rufus", "label": "ORG"},            # noise
                {"text": "Customer Service", "label": "ORG"}, # noise
                {"text": "Newest", "label": "ORG"},           # noise
            ],
            "embedding": e,
            "theme_cluster": 0,
            "app_version": "17.4",
            "thumbs_up": 0,
        }
        for i in range(40)  # need ≥ 3 mentions for entities to display
    ]
    issues = summ._score_issues(reviews)
    df = summ._aspect_doc_freq(reviews)
    issues_data = summ._issues_data(issues, df, len(reviews))

    surfaced = {e["text"] for e in issues_data[0]["entities"]}
    assert "Walmart" in surfaced, "legit entities still surface"
    for noise in ("Rufus", "Customer Service", "Newest"):
        assert noise not in surfaced, f"noise entity {noise!r} should be filtered"


# ---------------------------------------------------------------------------
# Phase X — trend attachment for leaderboard arrows
# ---------------------------------------------------------------------------

def test_attach_trends_marks_first_run_with_no_classification():
    """Without prior data, every issue gets trend.classification = None."""
    data = {
        "run_delta": {"first_run": True, "omitted": False},
        "issues": [{"cluster_id": 0}, {"cluster_id": 1}],
    }
    summ._attach_trends_to_issues(data)
    for issue in data["issues"]:
        assert issue["trend"] == {"classification": None, "pct_change": None}


def test_attach_trends_classifies_each_bucket():
    """Issues that appear in escalating / improving / new buckets get the
    matching classification; ones that appear in none default to 'stable'."""
    data = {
        "run_delta": {
            "first_run": False, "omitted": False, "prior_run_id": "x",
            "escalating": [{"cluster_id": 0, "pct_change": 50.0}],
            "improving":  [{"cluster_id": 1, "pct_change": -30.0}],
            "new":        [{"cluster_id": 2}],
            "resolved":   [],
        },
        "issues": [
            {"cluster_id": 0}, {"cluster_id": 1}, {"cluster_id": 2}, {"cluster_id": 3}
        ],
    }
    summ._attach_trends_to_issues(data)
    classifications = [i["trend"]["classification"] for i in data["issues"]]
    assert classifications == ["escalating", "improving", "new", "stable"]
    assert data["issues"][0]["trend"]["pct_change"] == 50.0


def test_attach_trends_omitted_run_delta_blank():
    """When snapshotting is off, all trends are None — leaderboard column blank."""
    data = {
        "run_delta": {"omitted": True},
        "issues": [{"cluster_id": 0}],
    }
    summ._attach_trends_to_issues(data)
    assert data["issues"][0]["trend"] == {"classification": None, "pct_change": None}


# ---------------------------------------------------------------------------
# _absa_section — loved / hated aggregation
# ---------------------------------------------------------------------------

def _absa(aspect, polarity):
    return {"aspect": aspect, "polarity": polarity, "confidence": 0.8}


def test_absa_section_groups_loved_and_hated_separately():
    """Aspects with consistently positive polarity → loved; consistently negative → hated."""
    reviews = []
    # 'design' loved (8 mentions @ +0.7)
    for _ in range(8):
        reviews.append({"aspects": [_absa("design", 0.7)]})
    # 'delivery' hated (10 mentions @ -0.6)
    for _ in range(10):
        reviews.append({"aspects": [_absa("delivery", -0.6)]})

    section = summ._absa_section(reviews)
    assert "Top Loved Features" in section
    assert "Top Hated Features" in section
    # design appears in loved, delivery in hated
    loved_idx = section.find("Top Loved Features")
    hated_idx = section.find("Top Hated Features")
    assert section.find("design", loved_idx, hated_idx) > -1
    assert section.find("delivery", hated_idx) > -1


def test_absa_section_suppresses_aspects_below_min_mentions():
    """Aspects with < ABSA_MIN_MENTIONS scored pairs are excluded — too noisy."""
    reviews = [
        {"aspects": [_absa("rare", -0.9)]},  # only 1 mention
        {"aspects": [_absa("rare", -0.9)]},
        {"aspects": [_absa("rare", -0.9)]},  # 3 < 5
    ]
    section = summ._absa_section(reviews)
    assert "rare" not in section


def test_absa_section_returns_empty_when_no_phase3_data():
    """If aspects field is still legacy list[str], no ABSA section."""
    reviews = [{"aspects": ["delivery", "package"]} for _ in range(20)]
    assert summ._absa_section(reviews) == ""


def test_absa_section_skips_neutral_aspects():
    """Aspects with avg_polarity=0 land in neither loved nor hated."""
    reviews = [{"aspects": [_absa("neutral_thing", 0.0)]} for _ in range(10)]
    section = summ._absa_section(reviews)
    # Neutral-only data → no section to render
    assert section == ""


# ---------------------------------------------------------------------------
# _build_snapshots
# ---------------------------------------------------------------------------

def _clustered_review(cid, rating=1, polarity=-0.5, urgency=0.6, embedding=None):
    return {
        "theme_cluster": cid,
        "rating": rating,
        "polarity": polarity,
        "urgency": urgency,
        "emotion": "anger",
        "aspects": ["delivery", "package", "driver"],
        "embedding": embedding if embedding is not None else [0.1, 0.2, 0.3],
        "subjectivity": 0.4,
    }


def test_build_snapshots_emits_one_per_cluster():
    """Issues *and* non-issues both get a snapshot row (D1: snapshot all clusters)."""
    issue_cluster = [_clustered_review(0, rating=1) for _ in range(10)]
    happy_cluster = [_clustered_review(1, rating=5, polarity=0.5) for _ in range(10)]
    reviews = issue_cluster + happy_cluster

    issues = summ._score_issues(reviews)
    df = summ._aspect_doc_freq(reviews)
    snapshots = summ._build_snapshots(reviews, issues, df, len(reviews))
    assert len(snapshots) == 2
    # Distinguishes is_issue=True vs False
    by_cid = {s["cluster_id"]: s for s in snapshots}
    assert by_cid[0]["is_issue"] is True
    assert by_cid[1]["is_issue"] is False


def test_build_snapshots_computes_centroid_from_embeddings():
    """Snapshot centroid = mean of cluster review embeddings."""
    reviews = [
        _clustered_review(0, embedding=[1.0, 0.0, 0.0]),
        _clustered_review(0, embedding=[0.0, 1.0, 0.0]),
        _clustered_review(0, embedding=[0.0, 0.0, 1.0]),
    ]
    issues = summ._score_issues(reviews)
    df = summ._aspect_doc_freq(reviews)
    snaps = summ._build_snapshots(reviews, issues, df, len(reviews))
    expected = [1 / 3, 1 / 3, 1 / 3]
    for got, want in zip(snaps[0]["centroid"], expected):
        assert abs(got - want) < 1e-6


def test_build_snapshots_priority_score_only_for_issues():
    """Non-issue clusters have priority_score=None; issues carry the float."""
    issue = [_clustered_review(0, rating=1) for _ in range(10)]
    happy = [_clustered_review(1, rating=5, polarity=0.5) for _ in range(10)]
    reviews = issue + happy
    issues = summ._score_issues(reviews)
    df = summ._aspect_doc_freq(reviews)
    snaps = summ._build_snapshots(reviews, issues, df, len(reviews))
    by_cid = {s["cluster_id"]: s for s in snaps}
    assert by_cid[0]["priority_score"] is not None
    assert by_cid[1]["priority_score"] is None


def test_build_snapshots_returns_empty_for_no_clusters():
    """Reviews with no theme_cluster set → no snapshots, no crash."""
    reviews = [{"theme_cluster": None, "embedding": [0.1, 0.2, 0.3]}]
    snaps = summ._build_snapshots(reviews, [], {}, len(reviews))
    assert snaps == []


# ---------------------------------------------------------------------------
# _run_delta_section — rendering branches
# ---------------------------------------------------------------------------

def _snap(cid, count, aspects=None, label="some label"):
    return {
        "cluster_id":     cid,
        "aspect_set":     list(aspects or ["delivery", "package"]),
        "cluster_label":  label,
        "review_count":   count,
        "centroid":       [1.0, 0.0],
    }


def test_run_delta_section_omitted_when_no_app_slug():
    """No persistence → no delta — the section disappears entirely."""
    out = summ._run_delta_section([], [], [], None, app_slug=None)
    assert out == ""


def test_run_delta_section_first_run_placeholder():
    """First run for an app shows the baseline placeholder."""
    out = summ._run_delta_section([], [], [], prior_run_id=None, app_slug="amazon")
    assert "baseline" in out


def test_run_delta_section_no_changes_branch():
    """Identical current/prior counts → 'no significant changes' message."""
    cur = [_snap(0, 100)]
    prior = [_snap(7, 100, aspects=["delivery", "package", "driver"])]
    # Match by Jaccard via cluster_label is ignored; aspect_set drives matching.
    # Force perfect aspect overlap so they match and are within ratio band.
    cur[0]["aspect_set"] = ["delivery", "package", "driver"]
    from pipeline.issue_tracking import match_issues, find_resolved
    matches = match_issues(cur, prior)
    resolved = find_resolved(matches, prior)
    out = summ._run_delta_section(matches, resolved, [], "20260101T000000Z", "amazon")
    assert "No significant changes" in out


def test_run_delta_section_renders_escalating_and_new():
    """Escalating + New + Resolved buckets render with their cluster labels."""
    cur = [
        _snap(0, 200, aspects=["delivery", "package", "driver"], label="Slow delivery"),
        _snap(1, 50, aspects=["fresh", "issue"], label="New cluster"),
    ]
    prior = [
        _snap(7, 100, aspects=["delivery", "package", "driver"], label="Delivery prev"),
        _snap(8, 80, aspects=["gone", "missing"], label="Resolved cluster"),
    ]
    from pipeline.issue_tracking import match_issues, find_resolved
    matches = match_issues(cur, prior)
    resolved = find_resolved(matches, prior)
    out = summ._run_delta_section(matches, resolved, [], "20260101T000000Z", "amazon")
    assert "Escalating" in out
    assert "New" in out
    assert "Resolved" in out
    # Labels surface in the output
    assert "Slow delivery" in out
    assert "Resolved cluster" in out
