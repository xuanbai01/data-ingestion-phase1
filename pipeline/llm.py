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
import json
import logging
import os
import re

from database.db import (
    load_issue_label,
    save_issue_label,
    load_takeaways,
    save_takeaways,
    load_section_narrative,
    save_section_narrative,
)

LLM_MODEL = "claude-haiku-4-5"
LLM_MAX_TOKENS = 50            # 4–6 word labels never need more than ~30 tokens
LLM_TEMPERATURE = 0.2          # low — we want deterministic-ish titles, not creativity
SAMPLE_REVIEW_MAX_CHARS = 200  # truncate long reviews so the prompt stays cheap

# Takeaways are 3-5 bullet points, each 1-2 sentences ≈ 30 tokens. Cap at
# 500 to give headroom without runaway output.
TAKEAWAYS_MAX_TOKENS = 500

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


# ---------------------------------------------------------------------------
# Key takeaways (Phase VIII — the "so what?" layer)
# ---------------------------------------------------------------------------

_TAKEAWAYS_PROMPT = """You write the executive takeaways for an app-store review analysis report. \
A non-technical reader (PM / engineer / leadership) reads this section to decide what to do.

Produce 3-5 bullet points. Each bullet must:
- Name a specific issue, feature, or pattern (cite Issue numbers, aspect names, app versions, or product names)
- Cite the supporting data inline (counts, percentages, ratings, deltas) — only numbers from the data below
- Imply an action: "fix first", "watch closely", "preserve", "investigate", etc.

Lead each bullet with a bold phrase using **markdown bold** — usually the recommended action or the entity being discussed. Keep each bullet 1-2 sentences. Be concrete; no vague claims. Do not invent numbers — only use what's in the data.

Output: a markdown bullet list (lines starting with "- "). No preamble, no closing remarks, no headings.

DATA:
{data_json}"""


def compute_takeaways_cache_key(synthesis_input):
    """Stable cache key for a takeaways generation, content-addressed by the
    synthesis input dict. Uses canonical JSON (sorted keys) so equivalent
    inputs produce the same key regardless of dict iteration order.
    """
    canonical = json.dumps(synthesis_input, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _build_synthesis_input(data):
    """Project the full report data dict down to just the fields the LLM needs
    for synthesis. Keeps the prompt under ~2000 tokens and avoids passing
    embeddings, raw review text, etc.
    """
    issues = data.get("issues") or []
    rd = data.get("run_delta") or {}
    absa = data.get("absa") or {}
    positives = (data.get("positives") or {}).get("entries") or []
    entities = data.get("entities") or []
    feature_summary = data.get("feature_summary") or {}
    overall = data.get("overall") or {}

    return {
        "corpus": {
            "n_reviews":   feature_summary.get("n_reviews"),
            "avg_rating":  round(overall.get("avg_rating", 0.0), 2),
            "neg_pct":     round(overall.get("neg_pct", 0.0), 2),
            "n_issues":    feature_summary.get("n_issues"),
        },
        "top_issues": [
            {
                "rank":        i["rank"],
                "label":       i["label"],
                "count":       i["count"],
                "avg_rating":  round(i["avg_rating"], 1),
                "score":       i["score"],
                "obj_share":   round(i["bug_complaint"]["obj_share"], 2),
                "top_emotions": [e["label"] for e in i.get("emotions", [])[:3]],
                "top_entities": [e["text"] for e in i.get("entities", [])[:3]],
                "top_versions": [v["version"] for v in i.get("app_versions", [])[:3]],
            }
            for i in issues
        ],
        "run_delta": {
            "first_run":  bool(rd.get("first_run")),
            "omitted":    bool(rd.get("omitted")),
            "escalating": [
                {"label": e["label"], "current": e["current_count"],
                 "prior": e["prior_count"], "pct_change": round(e["pct_change"], 0)}
                for e in (rd.get("escalating") or [])
            ],
            "improving":  [
                {"label": e["label"], "current": e["current_count"],
                 "prior": e["prior_count"], "pct_change": round(e["pct_change"], 0)}
                for e in (rd.get("improving") or [])
            ],
            "new":        [
                {"label": e["label"], "current": e["current_count"]}
                for e in (rd.get("new") or [])
            ],
            "resolved":   [
                {"label": e["label"], "prior": e["prior_count"]}
                for e in (rd.get("resolved") or [])
            ],
        },
        "absa_loved": [
            {"aspect": a["aspect"], "polarity": round(a["avg_polarity"], 2),
             "count": a["count"]}
            for a in (absa.get("loved") or [])[:5]
        ],
        "absa_hated": [
            {"aspect": a["aspect"], "polarity": round(a["avg_polarity"], 2),
             "count": a["count"]}
            for a in (absa.get("hated") or [])[:5]
        ],
        "positives": [
            {"label": p["label"], "count": p["count"],
             "avg_rating": round(p["avg_rating"], 1)}
            for p in positives[:5]
        ],
        "top_entities": [
            {"text": e["text"], "count": e["count"]}
            for e in entities[:5]
        ],
    }


def _parse_bullets(raw_text):
    """Extract bullet text from a markdown list. Tolerant of the LLM adding a
    short preamble or wrapping the list in extra whitespace. Returns the
    bullet text *with* the leading `- ` stripped.
    """
    if not raw_text:
        return []
    bullets = []
    for line in raw_text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Detect bullet lines: "- foo" or "* foo" or "1. foo" (defensive)
        m = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
        if m:
            bullets.append(m.group(1).strip())
        elif bullets:
            # Continuation line — append to the last bullet so 2-line bullets
            # don't get fragmented.
            bullets[-1] += " " + stripped
    return bullets


def generate_key_takeaways(report_data, use_cache=True):
    """
    Synthesize 3-5 executive takeaway bullets from the full report data.

    Returns list[str] of bullets (each a 1-2 sentence string with possible
    **markdown bold** for emphasis), or None on any failure (missing API
    key, network error, empty response, no data to synthesize).

    The caller is expected to render the bullets directly — this function
    keeps `**bold**` markers in the output so the markdown renderer can
    pass them through unchanged and the HTML renderer can convert via a
    Jinja filter.
    """
    if not (report_data and (report_data.get("issues") or report_data.get("run_delta"))):
        return None

    synthesis_input = _build_synthesis_input(report_data)
    cache_key = compute_takeaways_cache_key(synthesis_input)

    if use_cache:
        cached = load_takeaways(cache_key)
        if cached:
            return _parse_bullets(cached)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        LOG.warning(
            "ANTHROPIC_API_KEY not set; skipping LLM takeaways synthesis. "
            "Set the key for richer reports — falls back to omitting the section."
        )
        return None

    prompt = _TAKEAWAYS_PROMPT.format(
        data_json=json.dumps(synthesis_input, indent=2, default=str)
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=TAKEAWAYS_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            raise ValueError("empty content list in LLM response")
        raw = response.content[0].text
    except Exception as e:
        LOG.warning(f"LLM takeaways synthesis failed: {e}; section will be omitted")
        return None

    bullets = _parse_bullets(raw)
    if not bullets:
        LOG.warning("LLM returned no parseable bullets; section will be omitted")
        return None

    if use_cache:
        # Store the raw markdown so we can re-parse later if the parsing
        # logic changes; renderers re-parse on read anyway.
        save_takeaways(cache_key, raw, LLM_MODEL)

    return bullets


# ---------------------------------------------------------------------------
# Per-section narratives (Phase IX — one-sentence intros for kept detail sections)
# ---------------------------------------------------------------------------

# Tighter cap than takeaways — these are 1-2 sentence leads, not bullet lists.
NARRATIVE_MAX_TOKENS = 120

_NARRATIVE_PROMPT = """You write the lead sentence for one section of an app-review analysis report. \
A non-technical reader will scan this sentence to decide whether to read the section's data below.

Produce 1-2 sentences that:
- Name what the section shows (in plain English)
- Highlight the single most notable specific entry (with its name and the supporting number)
- Imply what the reader can do with the information ("preserve", "watch", "investigate", etc.) when relevant

If a specific action or named entity stands out, lead with it in **markdown bold**. Keep it short — this is a lead, not a paragraph. Use only numbers from the data; do not invent. No preamble, no closing remarks, no headings, no bullet markers. Output only the sentence.

SECTION: {section_name}
DATA:
{data_json}"""


def _compute_narrative_cache_key(section_name, data):
    """Cache key combines section name + canonical JSON of the section data.
    Different sections never collide; same data in the same section reuses."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    raw = f"{section_name}|{canonical}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _clean_narrative(text):
    """Strip whitespace, leading bullet markers, surrounding quotes."""
    if not text:
        return None
    cleaned = text.strip()
    # Drop a leading "- " or "* " in case the model added a bullet
    cleaned = re.sub(r"^[-*]\s+", "", cleaned)
    # Strip surrounding quotes
    cleaned = cleaned.strip('"').strip("'").strip()
    if not cleaned:
        return None
    # Cap at a sane maximum (defensive against runaway output)
    return cleaned[:600]


def generate_section_narrative(section_name, section_data, use_cache=True):
    """
    Synthesize a 1-2 sentence narrative lead for a detailed report section.

    Returns the narrative string (may contain `**markdown bold**`) or None
    on missing API key, API failure, empty response, or empty input data.
    Caller is expected to render the section without a lead on None.
    """
    if not section_data:
        return None

    cache_key = _compute_narrative_cache_key(section_name, section_data)

    if use_cache:
        cached = load_section_narrative(cache_key)
        if cached:
            return cached

    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Already logged once per run by the takeaways path; staying quiet
        # here avoids log spam from multiple narrative calls.
        return None

    prompt = _NARRATIVE_PROMPT.format(
        section_name=section_name,
        data_json=json.dumps(section_data, indent=2, default=str),
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=NARRATIVE_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            raise ValueError("empty content list in LLM response")
        raw = response.content[0].text
    except Exception as e:
        LOG.warning(f"LLM section narrative failed for {section_name}: {e}; section will render without lead")
        return None

    narrative = _clean_narrative(raw)
    if not narrative:
        return None

    if use_cache:
        save_section_narrative(cache_key, narrative, LLM_MODEL)
    return narrative
