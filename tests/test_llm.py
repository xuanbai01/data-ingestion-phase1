"""
Tests for pipeline/llm.py — the LLM cluster label seam.

The Anthropic client is mocked end-to-end so tests don't hit the network or
require an API key. We verify the four behaviors the rest of the pipeline
depends on:
- Cache hit short-circuits the API
- Cache miss writes a row that subsequent calls hit
- API exception → returns None (caller falls back) without raising
- Missing ANTHROPIC_API_KEY → returns None without trying to call
"""
import types

import pytest

from pipeline import llm
from database import db


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _FakeMessage:
    """Stand-in for anthropic's TextBlock content (just needs a .text attr)."""
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeMessage(text)]


class _FakeClient:
    """Minimal stand-in for anthropic.Anthropic(). Records calls for assertion."""

    def __init__(self, responses=None, raise_on_call=None):
        self._responses = list(responses or [])
        self._raise = raise_on_call
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        if not self._responses:
            return _FakeResponse("Default LLM label")
        return _FakeResponse(self._responses.pop(0))


def _patch_client(monkeypatch, client):
    """Replace `anthropic.Anthropic()` with a callable that returns our fake."""
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **kw: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


SAMPLE_ASPECTS = ["delivery", "package", "driver", "late"]
SAMPLE_HASHES = ["hash_a", "hash_b", "hash_c"]
SAMPLE_REVIEWS = [
    "Package arrived 5 days late and was damaged.",
    "Driver dropped the box on my doorstep without ringing.",
    "Order was supposed to come yesterday, still nothing.",
]


# ---------------------------------------------------------------------------
# compute_label_cache_key
# ---------------------------------------------------------------------------

def test_cache_key_stable_across_aspect_order():
    k1 = llm.compute_label_cache_key(["a", "b", "c"], SAMPLE_HASHES)
    k2 = llm.compute_label_cache_key(["c", "b", "a"], SAMPLE_HASHES)
    assert k1 == k2


def test_cache_key_stable_across_hash_order():
    k1 = llm.compute_label_cache_key(SAMPLE_ASPECTS, ["x", "y", "z"])
    k2 = llm.compute_label_cache_key(SAMPLE_ASPECTS, ["z", "y", "x"])
    assert k1 == k2


def test_cache_key_dedups_repeated_inputs():
    """Duplicates in either input shouldn't change the key."""
    k1 = llm.compute_label_cache_key(["a", "b"], ["x", "y"])
    k2 = llm.compute_label_cache_key(["a", "a", "b"], ["x", "y", "y"])
    assert k1 == k2


def test_cache_key_changes_when_inputs_differ():
    k1 = llm.compute_label_cache_key(["a", "b"], ["x"])
    k2 = llm.compute_label_cache_key(["a", "c"], ["x"])
    assert k1 != k2


# ---------------------------------------------------------------------------
# generate_cluster_label — happy path
# ---------------------------------------------------------------------------

def test_generate_label_happy_path_calls_api_and_caches(monkeypatch, temp_db):
    """First call hits the API; second call hits the cache (no API call)."""
    client = _FakeClient(responses=["Slow delivery and damaged packaging"])
    _patch_client(monkeypatch, client)

    label1 = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label1 == "Slow delivery and damaged packaging"
    assert len(client.calls) == 1

    # Second call with identical inputs must short-circuit on the cache.
    label2 = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label2 == label1
    assert len(client.calls) == 1, "second call should not hit the API"


def test_generate_label_strips_quotes_and_period(monkeypatch, temp_db):
    """Models sometimes wrap output in quotes or add a period — strip them."""
    client = _FakeClient(responses=['"Slow delivery and damaged packaging."'])
    _patch_client(monkeypatch, client)

    label = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label == "Slow delivery and damaged packaging"


def test_generate_label_passes_aspects_and_samples_into_prompt(monkeypatch, temp_db):
    """Sanity: the prompt actually contains the aspect words and review snippets."""
    client = _FakeClient(responses=["A reasonable title"])
    _patch_client(monkeypatch, client)

    llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    prompt = client.calls[0]["messages"][0]["content"]
    for aspect in SAMPLE_ASPECTS:
        assert aspect in prompt, f"aspect {aspect!r} missing from prompt"
    # At least one sample review snippet should appear (truncated)
    assert "Package arrived" in prompt


def test_generate_label_truncates_long_review_bodies(monkeypatch, temp_db):
    """A 5KB review must not blow up the prompt."""
    client = _FakeClient(responses=["Truncated title"])
    _patch_client(monkeypatch, client)

    huge = "A" * 5000
    llm.generate_cluster_label(SAMPLE_ASPECTS, [huge], SAMPLE_HASHES)
    prompt = client.calls[0]["messages"][0]["content"]
    # Truncated to SAMPLE_REVIEW_MAX_CHARS
    assert "A" * (llm.SAMPLE_REVIEW_MAX_CHARS + 1) not in prompt


# ---------------------------------------------------------------------------
# generate_cluster_label — failure paths
# ---------------------------------------------------------------------------

def test_generate_label_returns_none_when_api_key_missing(monkeypatch, temp_db):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    label = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label is None


def test_generate_label_returns_none_when_api_raises(monkeypatch, temp_db):
    """Network or auth errors must not propagate — caller relies on None to fall back."""
    client = _FakeClient(raise_on_call=RuntimeError("simulated 500"))
    _patch_client(monkeypatch, client)

    label = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label is None


def test_generate_label_returns_none_when_response_empty(monkeypatch, temp_db):
    """Whitespace-only responses get cleaned to None and shouldn't be cached."""
    client = _FakeClient(responses=["   "])
    _patch_client(monkeypatch, client)

    label = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label is None
    # Cache must NOT have a row — we don't want to memoize an empty response.
    cache_key = llm.compute_label_cache_key(SAMPLE_ASPECTS, SAMPLE_HASHES)
    assert db.load_issue_label(cache_key) is None


def test_generate_label_returns_none_for_empty_aspects(monkeypatch, temp_db):
    """Don't even try when there are no aspects to label."""
    client = _FakeClient()
    _patch_client(monkeypatch, client)

    label = llm.generate_cluster_label([], SAMPLE_REVIEWS, SAMPLE_HASHES)
    assert label is None
    assert client.calls == []


# ---------------------------------------------------------------------------
# Cache controls
# ---------------------------------------------------------------------------

def test_use_cache_false_bypasses_cache_read(monkeypatch, temp_db):
    """use_cache=False forces a fresh API call even when the row exists."""
    client = _FakeClient(responses=["First label", "Second label"])
    _patch_client(monkeypatch, client)

    first = llm.generate_cluster_label(SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES)
    second = llm.generate_cluster_label(
        SAMPLE_ASPECTS, SAMPLE_REVIEWS, SAMPLE_HASHES, use_cache=False
    )

    assert first == "First label"
    assert second == "Second label"
    assert len(client.calls) == 2


def test_save_load_round_trip(temp_db):
    """The label-cache helpers themselves round-trip cleanly."""
    db.save_issue_label("k1", "Some label", "claude-haiku-4-5")
    assert db.load_issue_label("k1") == "Some label"
    assert db.load_issue_label("nonexistent") is None


def test_save_replaces_on_duplicate_key(temp_db):
    """A second save under the same key overwrites — no duplicate-pk error."""
    db.save_issue_label("k1", "Old label", "claude-haiku-4-5")
    db.save_issue_label("k1", "New label", "claude-haiku-4-5")
    assert db.load_issue_label("k1") == "New label"


# ---------------------------------------------------------------------------
# Phase VIII — Key takeaways (LLM synthesis)
# ---------------------------------------------------------------------------

# Minimal report-data shape that's enough to exercise the synthesis path.
# Mirrors the data dict produced by build_report_data, but only the fields
# generate_key_takeaways() actually reads.
def _minimal_report_data():
    return {
        "issues": [
            {
                "rank":         1,
                "label":        "Slow delivery and damaged packages",
                "count":        100,
                "avg_rating":   1.4,
                "score":        0.74,
                "bug_complaint": {"obj_share": 0.62},
                "emotions":     [{"label": "anger", "share": 0.4}],
                "entities":     [{"text": "Walmart", "count": 10}],
                "app_versions": [{"version": "17.4", "count": 80}],
            }
        ],
        "run_delta": {
            "first_run":  False,
            "omitted":    False,
            "escalating": [
                {"label": "Issue 1", "current_count": 100, "prior_count": 70, "pct_change": 42.0}
            ],
            "improving": [],
            "new":       [],
            "resolved":  [],
        },
        "absa": {
            "loved": [{"aspect": "prime", "avg_polarity": 0.7, "count": 20}],
            "hated": [{"aspect": "login", "avg_polarity": -0.5, "count": 30}],
        },
        "positives":       {"entries": [{"label": "prime, membership", "count": 20, "avg_rating": 4.7}]},
        "entities":        [{"text": "Walmart", "count": 10}],
        "feature_summary": {"n_reviews": 100, "n_issues": 1},
        "overall":         {"avg_rating": 1.4, "neg_pct": 0.6},
    }


def test_takeaways_returns_bullets_on_happy_path(monkeypatch, temp_db):
    """The mocked LLM returns markdown bullets; generate_key_takeaways parses
    them into a list of strings."""
    raw_response = (
        "- **Fix Issue 1** because it grew 42% since last run.\n"
        "- **Investigate v17.4** — most mentions concentrate there.\n"
        "- **Preserve Prime** which is the strongest positive."
    )
    client = _FakeClient(responses=[raw_response])
    _patch_client(monkeypatch, client)

    bullets = llm.generate_key_takeaways(_minimal_report_data())
    assert bullets is not None
    assert len(bullets) == 3
    assert "**Fix Issue 1**" in bullets[0]


def test_takeaways_caches_on_repeat(monkeypatch, temp_db):
    """Second call with the same data should hit the cache — no API call."""
    client = _FakeClient(responses=["- **A** thing.\n- **B** thing."])
    _patch_client(monkeypatch, client)

    data = _minimal_report_data()
    first = llm.generate_key_takeaways(data)
    second = llm.generate_key_takeaways(data)

    assert first == second
    assert len(client.calls) == 1, "second call should hit cache, not API"


def test_takeaways_returns_none_without_api_key(monkeypatch, temp_db):
    """Missing ANTHROPIC_API_KEY → graceful skip (renderers omit section)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bullets = llm.generate_key_takeaways(_minimal_report_data())
    assert bullets is None


def test_takeaways_returns_none_when_api_raises(monkeypatch, temp_db):
    """Network / auth errors must not propagate — caller falls back to None."""
    client = _FakeClient(raise_on_call=RuntimeError("simulated 500"))
    _patch_client(monkeypatch, client)

    bullets = llm.generate_key_takeaways(_minimal_report_data())
    assert bullets is None


def test_takeaways_returns_none_on_empty_response(monkeypatch, temp_db):
    """If the LLM returns whitespace-only or no parseable bullets, skip."""
    client = _FakeClient(responses=["    \n    \n"])
    _patch_client(monkeypatch, client)

    bullets = llm.generate_key_takeaways(_minimal_report_data())
    assert bullets is None


def test_takeaways_returns_none_for_empty_data(monkeypatch, temp_db):
    """Don't call the API when there's nothing to synthesize."""
    client = _FakeClient()
    _patch_client(monkeypatch, client)

    bullets = llm.generate_key_takeaways({"issues": [], "run_delta": {}})
    assert bullets is None
    assert client.calls == []


def test_takeaways_cache_key_stable_across_dict_order(monkeypatch, temp_db):
    """Same content, different dict iteration order → same cache key."""
    a = {"a": 1, "b": [1, 2, 3]}
    b = {"b": [1, 2, 3], "a": 1}
    assert llm.compute_takeaways_cache_key(a) == llm.compute_takeaways_cache_key(b)


def test_parse_bullets_handles_continuation_lines():
    """Multi-line bullets (LLM wrapping its output) should be joined."""
    raw = (
        "- **First bullet** sentence one.\n"
        "  Sentence two on a continuation line.\n"
        "- **Second bullet** stands alone."
    )
    bullets = llm._parse_bullets(raw)
    assert len(bullets) == 2
    assert "Sentence two" in bullets[0]


def test_parse_bullets_handles_asterisk_and_numbered():
    """LLM might emit `* foo` or `1. foo` instead of `- foo`."""
    raw = (
        "* **A** asterisk.\n"
        "1. **B** numbered.\n"
        "- **C** dashed."
    )
    bullets = llm._parse_bullets(raw)
    assert len(bullets) == 3


def test_parse_bullets_drops_preamble():
    """If the LLM adds a preamble like 'Here are your takeaways:', drop it."""
    raw = (
        "Here are the key takeaways:\n"
        "- **Real** bullet.\n"
        "- **Another** bullet."
    )
    bullets = llm._parse_bullets(raw)
    assert len(bullets) == 2


def test_takeaways_save_load_round_trip(temp_db):
    """The takeaways-cache helpers themselves round-trip cleanly."""
    db.save_takeaways("k1", "- **Foo** bar.", "claude-haiku-4-5")
    assert db.load_takeaways("k1") == "- **Foo** bar."
    assert db.load_takeaways("nonexistent") is None


# ---------------------------------------------------------------------------
# Phase IX — Per-section narratives
# ---------------------------------------------------------------------------

def test_section_narrative_happy_path(monkeypatch, temp_db):
    """The mocked LLM returns a one-sentence lead; cleaned + cached + returned."""
    client = _FakeClient(responses=["**Prime membership** leads, with 200 reviews at 4.5★."])
    _patch_client(monkeypatch, client)

    out = llm.generate_section_narrative("positives", {"entries": [{"label": "prime", "count": 200}]})
    assert out is not None
    assert "Prime membership" in out


def test_section_narrative_caches(monkeypatch, temp_db):
    """Second call with same data short-circuits on cache."""
    client = _FakeClient(responses=["**A** thing."])
    _patch_client(monkeypatch, client)

    section_data = {"entries": [{"label": "x", "count": 1}]}
    first = llm.generate_section_narrative("positives", section_data)
    second = llm.generate_section_narrative("positives", section_data)
    assert first == second
    assert len(client.calls) == 1


def test_section_narrative_cache_keys_separated_by_section(monkeypatch, temp_db):
    """Same data passed under different section names → different cache keys.
    A 'positives' result must not be served when asking for 'absa'."""
    client = _FakeClient(responses=["**Positives** lead.", "**ABSA** says different."])
    _patch_client(monkeypatch, client)

    section_data = {"entries": [{"label": "x"}]}
    p = llm.generate_section_narrative("positives", section_data)
    a = llm.generate_section_narrative("absa", section_data)
    assert p != a
    assert len(client.calls) == 2


def test_section_narrative_returns_none_on_empty_data(monkeypatch, temp_db):
    """No data → no API call, no narrative."""
    client = _FakeClient()
    _patch_client(monkeypatch, client)

    assert llm.generate_section_narrative("positives", {}) is None
    assert llm.generate_section_narrative("positives", None) is None
    assert client.calls == []


def test_section_narrative_returns_none_on_api_failure(monkeypatch, temp_db):
    """API errors must not propagate — caller falls back to no narrative."""
    client = _FakeClient(raise_on_call=RuntimeError("simulated 500"))
    _patch_client(monkeypatch, client)

    out = llm.generate_section_narrative("positives", {"entries": [{"label": "x"}]})
    assert out is None


def test_section_narrative_returns_none_without_api_key(monkeypatch, temp_db):
    """Missing key → graceful skip (no warning spam — takeaways logs once)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = llm.generate_section_narrative("positives", {"entries": [{"label": "x"}]})
    assert out is None


def test_section_narrative_strips_leading_bullet(monkeypatch, temp_db):
    """If the LLM (against instructions) outputs a bullet marker, strip it."""
    client = _FakeClient(responses=["- **Action** is the lead."])
    _patch_client(monkeypatch, client)

    out = llm.generate_section_narrative("positives", {"entries": [{"label": "x"}]})
    assert not out.startswith("-")
    assert out.startswith("**Action**")


def test_section_narrative_save_load_round_trip(temp_db):
    """The section-narrative cache helpers themselves round-trip cleanly."""
    db.save_section_narrative("k1", "**Foo** bar.", "claude-haiku-4-5")
    assert db.load_section_narrative("k1") == "**Foo** bar."
    assert db.load_section_narrative("nonexistent") is None


def test_section_narrative_load_defensive_against_missing_table(temp_db):
    """Stale schemas (table doesn't exist) treat as cache miss, not crash."""
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("DROP TABLE IF EXISTS section_narratives_cache")
    conn.commit()
    conn.close()

    assert db.load_section_narrative("anything") is None  # no exception
