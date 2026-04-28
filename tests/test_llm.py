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
