"""
Tests for pipeline/summarizer.build_report_data — the data dict that drives
both the markdown and HTML renderers.

Focused on shape and key contents. The individual data extractors
(_score_issues, _build_snapshots, etc.) are tested separately in
test_summarizer.py; here we just verify the dict assembly.
"""
import os
os.environ.pop("ANTHROPIC_API_KEY", None)  # force LLM fallback during tests

import numpy as np
import pytest

from pipeline.summarizer import build_report_data


def _r(cluster_id, body, rating, polarity=-0.4, urgency=0.5, emotion="anger",
       aspects=("delivery", "package"), date="2026-01-01", embedding=None,
       version=None, thumbs=0, subjectivity=0.3):
    """Synthesize a feature-engineered review dict."""
    return {
        "reviewer_name": "x",
        "date":          date,
        "body":          body,
        "rating":        rating,
        "polarity":      polarity,
        "subjectivity":  subjectivity,
        "urgency":       urgency,
        "emotion":       emotion,
        "aspects":       [{"aspect": a, "polarity": -0.5, "confidence": 0.8} for a in aspects],
        "entities":      [{"text": "Google", "label": "ORG"}],
        "embedding":     embedding if embedding is not None else [0.1, 0.2, 0.3, 0.4],
        "theme_cluster": cluster_id,
        "app_version":   version,
        "thumbs_up":     thumbs,
    }


def _synthetic_corpus(n_negative=40, n_positive=20):
    """Two-cluster corpus: one negative (issue), one positive."""
    np.random.seed(0)
    e0 = (np.array([1, 0, 0, 0], dtype=np.float32) + np.random.randn(4) * 0.05).tolist()
    e1 = (np.array([0, 1, 0, 0], dtype=np.float32) + np.random.randn(4) * 0.05).tolist()

    reviews = []
    for i in range(n_negative):
        reviews.append(_r(0, "late delivery and broken box", 1,
                          aspects=("delivery", "package", "driver", "late"),
                          date=f"2026-01-{(i % 28) + 1:02d}",
                          embedding=e0, version="17.4", thumbs=i))
    for i in range(n_positive):
        reviews.append(_r(1, "amazon prime is great love it", 5, polarity=0.5,
                          urgency=0.1, emotion="joy",
                          aspects=("prime", "membership"),
                          date=f"2026-03-{(i % 28) + 1:02d}",
                          embedding=e1))
    return reviews


# ---------------------------------------------------------------------------
# Top-level dict shape
# ---------------------------------------------------------------------------

EXPECTED_TOP_KEYS = {
    "header", "overall", "issues", "run_delta", "positives", "absa",
    "urgent", "emotions", "entities", "aspect_index", "feature_summary",
    "_meta",
}


def test_build_report_data_returns_all_top_level_sections():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    assert set(data.keys()) == EXPECTED_TOP_KEYS


def test_build_report_data_no_app_slug_means_run_delta_omitted():
    """Without app_slug, the snapshot/match path is skipped and run_delta says 'omitted'."""
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", app_slug=None, persist_snapshots=False)
    assert data["run_delta"] == {"omitted": True}


def test_meta_carries_app_name_and_run_id():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    assert data["_meta"]["app_name"] == "TestApp"
    # run_id is an ISO compact timestamp like "20260428T013817Z"
    assert len(data["_meta"]["run_id"]) >= 16
    assert data["_meta"]["run_id"].endswith("Z")


# ---------------------------------------------------------------------------
# Header + overall
# ---------------------------------------------------------------------------

def test_header_has_review_count_and_app_name():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    h = data["header"]
    assert h["app_name"] == "TestApp"
    assert h["review_count"] == len(reviews)
    assert h["generated_at"]
    assert h["run_id"]


def test_overall_polarity_buckets_sum_to_total_with_polarity():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    o = data["overall"]
    assert o["pos_count"] + o["neutral_count"] + o["neg_count"] == o["n_with_polarity"]


# ---------------------------------------------------------------------------
# Issues — full per-card data
# ---------------------------------------------------------------------------

def test_issues_have_required_card_fields():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    assert data["issues"], "expected at least one priority issue"
    issue = data["issues"][0]
    required = {
        "rank", "cluster_id", "label", "aspect_string", "label_is_llm",
        "count", "avg_rating", "avg_urgency", "avg_polarity", "score",
        "components", "bug_complaint", "emotions", "entities",
        "representative_reviews", "sparkline", "app_versions",
    }
    assert required.issubset(issue.keys())


def test_issue_label_falls_back_to_aspect_string_without_llm():
    """No ANTHROPIC_API_KEY → label_is_llm=False, label = aspect_string."""
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    issue = data["issues"][0]
    assert issue["label_is_llm"] is False
    assert issue["label"] == issue["aspect_string"]


def test_issue_sparkline_buckets_sum_to_count_when_dates_span_window():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    issue = data["issues"][0]
    if issue["sparkline"]:
        assert sum(issue["sparkline"]["buckets"]) == issue["count"]
        assert len(issue["sparkline"]["buckets"]) == 12


def test_issue_app_versions_aggregated_from_cluster():
    """App-version counts should reflect the cluster's review version field."""
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    issue = data["issues"][0]
    # The negative cluster all has version="17.4" → exactly one entry, count == 40
    versions = {v["version"]: v["count"] for v in issue["app_versions"]}
    assert versions.get("17.4") == 40


def test_issue_emotions_share_sums_to_one():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    issue = data["issues"][0]
    if issue["emotions"]:
        total_share = sum(e["share"] for e in issue["emotions"])
        assert abs(total_share - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# ABSA / urgent / entities / aspect_index — basic shape
# ---------------------------------------------------------------------------

def test_urgent_entries_are_dicts_with_review_fields():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    if data["urgent"]:
        u = data["urgent"][0]
        assert {"urgency", "rating", "emotion", "body"}.issubset(u.keys())


def test_emotions_entries_have_emotion_count_share():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    e = data["emotions"]
    assert "entries" in e and "total" in e
    if e["entries"]:
        assert {"emotion", "count", "share"}.issubset(e["entries"][0].keys())


def test_positives_entries_renamed_to_avoid_jinja_collision():
    """Both positives and emotions use 'entries' (not 'items') to avoid the
    Jinja attribute-vs-method collision with dict.items().
    """
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    assert "entries" in data["positives"]
    assert "items" not in data["positives"]


def test_feature_summary_counts_match_inputs():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    fs = data["feature_summary"]
    assert fs["n_reviews"] == len(reviews)
    assert fs["themes"] == 2  # two clusters in the corpus
    assert fs["n_issues"] == 1  # only the negative cluster qualifies


# ---------------------------------------------------------------------------
# HTML rendering — smoke
# ---------------------------------------------------------------------------

def test_render_html_returns_self_contained_html():
    """End-to-end HTML render: the document is well-formed and embeds assets."""
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    html = render_html(data)

    assert html is not None, "render_html should produce output when assets exist"
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # Pico CSS should be embedded (its license header is recognizable)
    assert "Pico CSS" in html or "@charset" in html
    # Chart.js should be embedded — ~100KB+
    assert len(html) > 200_000, f"expected embedded assets, got {len(html)} bytes"
    # Each issue gets a sparkline canvas
    if any(i["sparkline"] for i in data["issues"]):
        assert "data-sparkline" in html


def test_render_html_includes_issue_labels_and_subtitles():
    """Issue heading, aspect subtitle, and stats should all surface in the HTML."""
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    html = render_html(data)
    issue = data["issues"][0]
    assert f"Issue {issue['rank']}" in html
    # Aspect string surfaces somewhere — either in heading (no LLM) or subtitle (LLM)
    for asp in issue["aspect_string"].split(", "):
        if asp:
            assert asp in html


def test_render_html_handles_no_run_delta_section():
    """When app_slug is None, the Run Delta section is omitted entirely.
    Test against the actual heading element, not the HTML comment that
    documents the section's place in the template.
    """
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", app_slug=None, persist_snapshots=False)
    html = render_html(data)
    # Section heading element should not appear
    assert '<section id="delta">' not in html
    assert "<h2>Run Delta</h2>" not in html


def test_render_html_runs_first_run_branch_with_app_slug():
    """With app_slug set but no prior snapshots, the placeholder text appears."""
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    # app_slug set but persist_snapshots=False → first_run=True branch in run_delta
    data = build_report_data(reviews, "TestApp", app_slug="testapp",
                             persist_snapshots=False)
    html = render_html(data)
    assert "Run Delta" in html
    assert "baseline" in html.lower()
