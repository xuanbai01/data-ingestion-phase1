"""
Cross-run issue matching for Phase IV (time dimension).

Given a current run's cluster snapshots and a prior run's snapshots, decide
which current issue continues which prior issue. Aspect Jaccard is the
primary signal — interpretable, robust to small wording shifts, and naturally
handles the case where the cluster grew or shrank but its core stayed the
same. Embedding centroid cosine is the fallback for "soft" cases where the
aspect words drifted but the semantic theme is the same.

Pure functions, no DB access, no model calls — easy to unit-test on
synthetic snapshots.
"""
import numpy as np


# Thresholds — picked from the handoff. Loose enough to absorb run-to-run
# noise in aspect extraction, tight enough that unrelated clusters don't
# falsely merge.
JACCARD_THRESHOLD = 0.4
CENTROID_THRESHOLD = 0.7

# Run-delta classification bands. Outside [IMPROVE, ESCALATE] we flag the
# direction; inside, the issue is "stable" and excluded from the delta view.
ESCALATE_RATIO = 1.2
IMPROVE_RATIO = 0.8


def jaccard(a, b):
    """|A ∩ B| / |A ∪ B|. 0.0 when both sets are empty."""
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def cosine(a, b):
    """Cosine similarity between two embedding vectors. 0.0 if either is None / zero-norm."""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def match_issues(current, prior,
                 jaccard_threshold=JACCARD_THRESHOLD,
                 centroid_threshold=CENTROID_THRESHOLD):
    """
    For each current snapshot, find the best prior match. Returns a list
    aligned with `current`, one dict per current snapshot:

        {
            "current": <current snapshot>,
            "prior":   <prior snapshot or None>,
            "method":  "jaccard" | "centroid" | None,
            "score":   float (the matching score),
        }

    Many-to-one matches (a cluster split — two current issues mapping to the
    same prior) are allowed: each current independently picks its best prior.
    Resolution detection (priors with no current match) is exposed via
    `find_resolved` since callers usually want both views.
    """
    matches = []
    for c in current:
        # Primary: Jaccard on aspect sets
        best = (None, 0.0)
        for p in prior:
            s = jaccard(c.get("aspect_set"), p.get("aspect_set"))
            if s > best[1]:
                best = (p, s)

        if best[1] >= jaccard_threshold:
            matches.append({
                "current": c,
                "prior":   best[0],
                "method":  "jaccard",
                "score":   round(best[1], 3),
            })
            continue

        # Fallback: centroid cosine. Skip if current has no centroid (rare —
        # would only happen if every review in the cluster lacked an embedding).
        c_centroid = c.get("centroid")
        if c_centroid is None:
            matches.append({"current": c, "prior": None, "method": None, "score": 0.0})
            continue

        best_centroid = (None, 0.0)
        for p in prior:
            s = cosine(c_centroid, p.get("centroid"))
            if s > best_centroid[1]:
                best_centroid = (p, s)

        if best_centroid[1] >= centroid_threshold:
            matches.append({
                "current": c,
                "prior":   best_centroid[0],
                "method":  "centroid",
                "score":   round(best_centroid[1], 3),
            })
        else:
            matches.append({"current": c, "prior": None, "method": None, "score": 0.0})

    return matches


def find_resolved(matches, prior):
    """
    Prior snapshots that no current snapshot matched — candidates for the
    "resolved" delta bucket. Match keys on cluster_id since it's unique
    within a single prior run.
    """
    matched_ids = {
        m["prior"]["cluster_id"]
        for m in matches
        if m["prior"] is not None
    }
    return [p for p in prior if p["cluster_id"] not in matched_ids]


def classify_delta(match,
                   escalate_ratio=ESCALATE_RATIO,
                   improve_ratio=IMPROVE_RATIO):
    """
    Tag a match as 'new', 'escalating', 'improving', or 'stable'.

    'new' is asymmetric with 'resolved': new is a current issue with no prior
    match, resolved is a prior issue with no current match. Resolved isn't
    classified here — it's its own bucket from `find_resolved`.
    """
    if match["prior"] is None:
        return "new"
    prior_count = match["prior"].get("review_count") or 0
    if prior_count == 0:
        return "new"
    ratio = match["current"].get("review_count", 0) / prior_count
    if ratio > escalate_ratio:
        return "escalating"
    if ratio < improve_ratio:
        return "improving"
    return "stable"


# ---------------------------------------------------------------------------
# ASCII sparklines
# ---------------------------------------------------------------------------

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def render_sparkline(values):
    """Render a list of numeric values as an 8-level Unicode block sparkline.

    Falls through to "—" when the list is empty or all-None. A flat (zero-span)
    series renders as the lowest bar across — visually distinct from "no data".
    None entries within the series render as a space (a gap in the line).
    """
    if not values:
        return "—"
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return "—"
    lo = min(cleaned)
    hi = max(cleaned)
    span = hi - lo
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        elif span == 0:
            out.append(SPARK_BLOCKS[0])
        else:
            idx = int((v - lo) / span * (len(SPARK_BLOCKS) - 1))
            idx = max(0, min(len(SPARK_BLOCKS) - 1, idx))
            out.append(SPARK_BLOCKS[idx])
    return "".join(out)


def bucket_dates(date_strs, n_buckets=12):
    """Bucket a list of YYYY-MM-DD strings into `n_buckets` equal-time slots
    spanning [earliest, latest]. Returns (counts list[int], earliest_str,
    latest_str) or (None, None, None) if there aren't enough valid dates.

    Used for per-issue intra-run sparklines.
    """
    from datetime import datetime
    parsed = []
    for d in date_strs or []:
        if not d:
            continue
        try:
            parsed.append(datetime.strptime(d, "%Y-%m-%d"))
        except (ValueError, TypeError):
            continue
    if len(parsed) < 2:
        return None, None, None

    earliest = min(parsed)
    latest = max(parsed)
    span_days = (latest - earliest).days
    if span_days <= 0:
        # All reviews on the same day — single-bar sparkline isn't informative.
        return None, None, None

    bucket = span_days / n_buckets
    counts = [0] * n_buckets
    for d in parsed:
        # Clamp the last day into the final bucket (avoid off-by-one at the boundary).
        idx = min(int((d - earliest).days / bucket), n_buckets - 1)
        counts[idx] += 1

    return counts, earliest.strftime("%Y-%m-%d"), latest.strftime("%Y-%m-%d")
