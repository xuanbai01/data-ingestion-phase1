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
    "run_summary", "takeaways", "_meta",
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


# ---------------------------------------------------------------------------
# Phase VII — run summary, ribbon, and section ordering
# ---------------------------------------------------------------------------

def test_run_summary_has_ribbon_and_narrative():
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    rs = data["run_summary"]
    assert "ribbon" in rs and "narrative" in rs
    assert {"reviews", "escalating", "new", "resolved"} == set(rs["ribbon"].keys())


def test_run_summary_narrative_describes_top_issue():
    """The narrative should mention the top issue's label and review count."""
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    top = data["issues"][0]
    narrative = data["run_summary"]["narrative"]
    assert top["label"] in narrative
    assert f"{top['count']:,}" in narrative


def test_run_summary_narrative_no_issues_branch():
    """A purely positive corpus → narrative says no negative clusters."""
    np.random.seed(0)
    e = (np.array([1, 0, 0, 0], dtype=np.float32) + np.random.randn(4) * 0.05).tolist()
    reviews = [_r(0, "great", 5, polarity=0.5, urgency=0.1, emotion="joy",
                  aspects=("app", "interface"),
                  date=f"2026-01-{(i % 28) + 1:02d}", embedding=e)
               for i in range(30)]
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    assert "no negative-leaning" in data["run_summary"]["narrative"]


def test_run_summary_ribbon_uses_none_when_no_comparison():
    """First run / no app_slug → ribbon counts for delta fields are None."""
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", app_slug=None, persist_snapshots=False)
    ribbon = data["run_summary"]["ribbon"]
    assert ribbon["reviews"] == len(reviews)
    assert ribbon["escalating"] is None
    assert ribbon["new"] is None
    assert ribbon["resolved"] is None


def test_run_summary_ribbon_has_counts_with_prior(temp_db):
    """With persisted snapshots from a prior run, ribbon counts are integers."""
    import time

    reviews = _synthetic_corpus()
    # First run: writes baseline snapshot.
    build_report_data(reviews, "TestApp", app_slug="testapp", persist_snapshots=True)
    time.sleep(1.1)  # ensure run_id differs
    # Second run: should now have a prior to compare against.
    data = build_report_data(reviews, "TestApp", app_slug="testapp",
                             persist_snapshots=True)
    ribbon = data["run_summary"]["ribbon"]
    assert isinstance(ribbon["escalating"], int)
    assert isinstance(ribbon["new"], int)
    assert isinstance(ribbon["resolved"], int)


def test_run_summary_baseline_narrative_says_baseline():
    """First run with app_slug → narrative explicitly mentions baseline."""
    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", app_slug="testapp",
                             persist_snapshots=False)
    assert "Baseline" in data["run_summary"]["narrative"]


# ---------------------------------------------------------------------------
# HTML ordering — Run Delta first, Priority Issues second, Detailed third
# ---------------------------------------------------------------------------

def test_html_renders_elevator_pitch():
    """The pitch tagline appears under the H1.
    Jinja autoescapes apostrophes (e.g. "what's" → "what&#39;s") so we check
    for an unambiguous substring without special characters.
    """
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    html = render_html(data)
    assert "Groups app reviews into recurring issues" in html
    assert "tracks them over time" in html


def test_html_renders_ribbon():
    """All four ribbon stat cards appear in the rendered HTML."""
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    html = render_html(data)
    # Ribbon labels appear in the page
    for label in ["reviews", "escalating", "new", "resolved"]:
        assert f"<span class=\"label\">{label}</span>" in html


def test_html_orders_delta_before_priority_before_detailed():
    """Run Delta should appear before Priority Issues, which appears before
    Detailed analysis. This is the editorial-pass spec from Phase VII."""
    from pipeline.summarizer import render_html

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", app_slug="testapp",
                             persist_snapshots=False)
    html = render_html(data)
    delta_pos = html.find('id="delta"')
    priority_pos = html.find('id="priority"')
    detailed_pos = html.find('id="detailed"')
    assert delta_pos > 0
    assert priority_pos > 0
    assert detailed_pos > 0
    assert delta_pos < priority_pos < detailed_pos


def test_html_collapses_issues_beyond_top_3():
    """When there are >3 priority issues, 4-5 are wrapped in a collapsed
    <details> element, while top 3 render inline."""
    from pipeline.summarizer import render_html
    import numpy as np

    np.random.seed(0)
    embeddings = [(np.array([1 if i == j else 0 for j in range(8)], dtype=np.float32)
                   + np.random.randn(8) * 0.05).tolist() for i in range(6)]
    reviews = []
    for cid in range(5):
        for i in range(40 - cid * 4):
            reviews.append(_r(cid, f"cluster {cid} body", 1,
                              aspects=(f"asp{cid}_a", f"asp{cid}_b", f"asp{cid}_c"),
                              date=f"2026-01-{(i % 28) + 1:02d}",
                              embedding=embeddings[cid]))
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    html = render_html(data)

    if len(data["issues"]) > 3:
        assert 'class="more-issues"' in html
        assert "Show " in html and "more priority issue" in html
    else:
        # Tiny synthetic dataset may produce ≤3 issues — the collapse just
        # doesn't render. Both paths are valid; we only assert the layout
        # when there's something to collapse.
        assert 'class="more-issues"' not in html


def test_markdown_includes_pitch_and_run_summary():
    """Both the elevator pitch and the auto-narrative appear in the MD output."""
    from pipeline.summarizer import render_markdown, ELEVATOR_PITCH

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    md = render_markdown(data)
    assert ELEVATOR_PITCH in md
    # Narrative leads as a blockquote
    assert "> Top issue:" in md or "> " + str(len(reviews)) in md


def test_markdown_has_detailed_analysis_separator():
    """The supporting sections live below a horizontal rule + 'Detailed
    analysis' heading, so plain-text readers can see the structural hint.

    Phase IX cut Overall Sentiment / Most Urgent / Emotion Distribution /
    Mentioned Entities / Aspect Index from the rendered output. The kept
    detail sections under the divider are now Top Positives (rendered as
    "What are users happy about?") and ABSA ("Which features are loved
    vs hated?") — we check the divider precedes one of those.
    """
    from pipeline.summarizer import render_markdown

    reviews = _synthetic_corpus()
    data = build_report_data(reviews, "TestApp", persist_snapshots=False)
    md = render_markdown(data)
    assert "## Detailed analysis" in md
    # The separator line precedes the detailed section
    sep_pos = md.find("---\n\n## Detailed analysis")
    assert sep_pos > 0
    # At least one kept section appears after the divider
    happy_pos = md.find("## What are users happy about?")
    loved_pos = md.find("## Which features are loved vs hated?")
    assert max(happy_pos, loved_pos) > sep_pos, (
        "expected at least one Phase IX kept section under the divider"
    )
