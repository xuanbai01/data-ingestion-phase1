"""
LLM cluster labeling (Phase V — A1).

A single seam between the pipeline and any chat-completion LLM. Currently
configured for Anthropic / Claude Haiku. To swap providers later (DeepSeek
via their Anthropic-format endpoint, for example), only `_call_llm` needs
to change.

Design points:

* The label cache is content-addressed by md5(sorted(review_hashes) +
  sorted(aspect_set)). Same cluster contents → same cache key → no extra
  API call on a re-run. Cache key is independent of cluster_id (which is
  arbitrary KMeans output) and run_id (which always changes), so it
  meaningfully reuses across runs of the same data.

* Failures (no API key, network error, malformed response) return None and
  log a warning. The caller is expected to fall back to its existing
  aspect-string label on None — never crash the pipeline because the LLM
  is down.

* No retries. Cluster labels are nice-to-have polish; if the API hiccups,
  one run gets aspect-string fallback labels and the next run will retry
  on the same cache miss. Keeps the code simple.
"""
import hashlib
import logging
import os

from database.db import load_issue_label, save_issue_label

LLM_MODEL = "claude-haiku-4-5"
LLM_MAX_TOKENS = 50            # 4–6 word labels never need more than ~30 tokens
LLM_TEMPERATURE = 0.2          # low — we want deterministic-ish titles, not creativity
SAMPLE_REVIEW_MAX_CHARS = 200  # truncate long reviews so the prompt stays cheap

LOG = logging.getLogger("pipeline")


def compute_label_cache_key(aspects, review_hashes):
    """Stable cache key for a cluster's label.

    Both inputs are sorted before hashing so set order doesn't affect the key.
    Uses md5 (same hash family already used elsewhere in the project — same
    32-char hex) for cheap, collision-irrelevant uniqueness on small inputs.
    """
    aspect_part = ",".join(sorted(set(aspects or [])))
    hash_part = ",".join(sorted(set(review_hashes or [])))
    raw = f"{aspect_part}|{hash_part}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_prompt(aspects, sample_reviews):
    """Tight prompt — one task, no preamble, low risk of injection given the truncation."""
    aspect_str = ", ".join(aspects)
    samples = []
    for i, body in enumerate(sample_reviews or [], 1):
        if not body:
            continue
        clean = body.replace("\n", " ").strip()[:SAMPLE_REVIEW_MAX_CHARS]
        samples.append(f"{i}. {clean}")
    sample_block = "\n".join(samples) if samples else "(no sample reviews)"
    return (
        "You write concise titles for clusters of app-store reviews.\n"
        "Output a 4-6 word title that names the central theme of these reviews.\n"
        "No quotes, no period, no preamble. Output only the title.\n"
        "\n"
        f"Aspects: {aspect_str}\n"
        "\n"
        "Sample reviews:\n"
        f"{sample_block}"
    )


def _call_llm(prompt):
    """Talk to the configured chat LLM. Returns the raw text response, or
    raises an exception on any failure (caller catches and falls back).

    This is the only function that touches a provider SDK — swap providers
    by changing this body.
    """
    # Imported lazily so test environments without the SDK can still load
    # the module (and so the 200ms anthropic import doesn't pay every run
    # if the cache is fully warm — generate_cluster_label short-circuits
    # before this is called).
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    # response.content is a list of content blocks; for a plain text response
    # there is one TextBlock at index 0.
    if not response.content:
        raise ValueError("empty content list in LLM response")
    return response.content[0].text


def _clean_label(text):
    """Strip whitespace, surrounding quotes, trailing period; cap at ~80 chars.

    Models sometimes wrap output in quotes despite instructions — strip those.
    Trailing period is occasional and not wanted in a heading. Cap length as
    a defensive guard against runaway output.
    """
    if not text:
        return None
    cleaned = text.strip().strip('"').strip("'").strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        return None
    return cleaned[:80]


def generate_cluster_label(aspects, sample_reviews, review_hashes, use_cache=True):
    """
    Return a 4-6 word title for a review cluster, or None if the LLM is
    unavailable / errored / returned an empty response.

    Caller is expected to fall back to an aspect-string label on None.
    """
    if not aspects:
        return None

    cache_key = compute_label_cache_key(aspects, review_hashes)

    if use_cache:
        cached = load_issue_label(cache_key)
        if cached:
            return cached

    if not os.environ.get("ANTHROPIC_API_KEY"):
        LOG.warning(
            "ANTHROPIC_API_KEY not set; skipping LLM cluster labeling and "
            "falling back to aspect-string labels."
        )
        return None

    prompt = _build_prompt(aspects, sample_reviews)
    try:
        raw = _call_llm(prompt)
    except Exception as e:
        LOG.warning(f"LLM cluster labeling failed: {e}; falling back to aspect-string label")
        return None

    label = _clean_label(raw)
    if not label:
        LOG.warning("LLM returned empty label after cleaning; falling back")
        return None

    if use_cache:
        save_issue_label(cache_key, label, LLM_MODEL)
    return label
