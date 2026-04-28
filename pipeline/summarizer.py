"""
Summarizer: consumes feature-engineered reviews and produces a human-readable
markdown report plus a short terminal summary.

The report is organized around "issues" — negative-leaning theme clusters
ranked by a composite priority score. Aspects, subjectivity, and emotion
appear as drill-down inside each issue rather than as parallel sections.
"""
import os
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np

from database.db import (
    save_issue_snapshots,
    load_prior_run_snapshots,
    compute_cache_key,
)
from pipeline.issue_tracking import (
    match_issues,
    find_resolved,
    classify_delta,
    bucket_dates,
    render_sparkline,
)
from pipeline.llm import generate_cluster_label

REPORT_DIR = "reports"

# One-sentence pitch that anchors the report. Renders as the subtitle on
# the HTML page and the lead line of the markdown report. Resist the urge
# to mention any technique here — see Phase VII feedback rationale.
ELEVATOR_PITCH = (
    "Groups app reviews into recurring issues and tracks them over time — "
    "surfacing what's getting worse, improving, new, or resolved."
)

NEG_THRESHOLD = -0.1
POS_THRESHOLD = 0.1
SUBJECTIVE_THRESHOLD = 0.5

# A cluster qualifies as an "issue" if it skews negative on either rating or polarity.
ISSUE_RATING_CEILING = 3.0
ISSUE_POLARITY_CEILING = -0.05

# Priority score weights (sum to 1.0). Volume + severity dominate;
# urgency and emotion intensity sharpen the ranking among similar-volume clusters.
W_VOLUME = 0.30
W_SEVERITY = 0.30
W_URGENCY = 0.25
W_EMOTION = 0.15

INTENSE_EMOTIONS = {"anger", "disgust", "fear"}

TOP_ISSUES_N = 5
TOP_POSITIVES_N = 5

# TF-IDF aspect labels are unstable on tiny clusters; fall back to raw frequency.
TFIDF_MIN_CLUSTER_SIZE = 20

# ABSA: minimum mentions for an aspect to appear in loved/hated tables.
ABSA_MIN_MENTIONS = 5


def _aspect_names(review):
    """Extract aspect name strings from a review dict.
    Handles both list[{aspect, polarity, confidence}] (Phase 3+) and list[str].
    """
    return [
        a["aspect"] if isinstance(a, dict) else a
        for a in (review.get("aspects") or [])
    ]


# ---------------------------------------------------------------------------
# Report data assembly (Phase VI)
#
# `build_report_data` is the single source of truth for everything the
# markdown and HTML renderers consume. Each `_xxx_data` function returns
# the structured data for one section; renderers do formatting only.
# ---------------------------------------------------------------------------

def build_report_data(reviews, app_name, app_slug=None, run_id=None,
                      persist_snapshots=True):
    """
    Compute every piece of data the markdown and HTML renderers need.

    Side effects (when `persist_snapshots=True` and `app_slug` is set):
      - writes this run's per-cluster snapshots to issue_snapshots
      - reads the most recent prior run's snapshots for cross-run matching

    Pass `persist_snapshots=False` to compute the data without touching the
    DB (useful for tests and previews).
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    issues = _score_issues(reviews)
    corpus_df = _aspect_doc_freq(reviews)
    total = len(reviews)

    # LLM labels — silent fallback to aspect strings on missing key / API error.
    _attach_llm_labels(issues, corpus_df, total)

    # Snapshots for every cluster. Generated regardless of persistence so the
    # data dict shape stays consistent across modes.
    all_snapshots = _build_snapshots(reviews, issues, corpus_df, total)

    matches = []
    resolved = []
    prior_run_id = None
    if persist_snapshots and app_slug and all_snapshots:
        save_issue_snapshots(app_slug, run_id, all_snapshots)
        prior_run_id, prior_snapshots = load_prior_run_snapshots(app_slug, run_id)
        if prior_snapshots:
            matches = match_issues(all_snapshots, prior_snapshots)
            resolved = find_resolved(matches, prior_snapshots)

    data = {
        "header":          _header_data(reviews, app_name, run_id),
        "overall":         _overall_sentiment_data(reviews),
        "issues":          _issues_data(issues, corpus_df, total),
        "run_delta":       _run_delta_data(matches, resolved, issues, prior_run_id, app_slug),
        "positives":       _positives_data(reviews, corpus_df),
        "absa":            _absa_data(reviews),
        "urgent":          _urgent_data(reviews),
        "emotions":        _emotion_data(reviews),
        "entities":        _entities_data(reviews),
        "aspect_index":    _aspect_index_data(reviews, issues),
        "feature_summary": _feature_summary_data(reviews, issues),
        "_meta": {
            "app_name":     app_name,
            "app_slug":     app_slug,
            "run_id":       run_id,
            "issues_full":  issues,  # full unfiltered list (for terminal summary)
        },
    }
    # Run summary is derived from issues + run_delta + feature_summary, so it
    # has to be computed after the rest of the dict is assembled.
    data["run_summary"] = _run_summary_data(data)
    return data


# --- Per-section data extractors -----------------------------------------

def _header_data(reviews, app_name, run_id):
    return {
        "app_name":     app_name,
        "run_id":       run_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "review_count": len(reviews),
    }


def _overall_sentiment_data(reviews):
    rated = [r for r in reviews if r.get("rating") is not None]
    with_pol = [r for r in reviews if r.get("polarity") is not None]
    n = len(with_pol) or 1

    avg_rating = float(np.mean([r["rating"] for r in rated])) if rated else 0.0
    avg_pol = float(np.mean([r["polarity"] for r in with_pol])) if with_pol else 0.0

    neg = [r for r in with_pol if r["polarity"] < NEG_THRESHOLD]
    pos = [r for r in with_pol if r["polarity"] > POS_THRESHOLD]
    obj_neg = [r for r in neg if (r.get("subjectivity") or 0) < SUBJECTIVE_THRESHOLD]
    subj_neg = [r for r in neg if (r.get("subjectivity") or 0) >= SUBJECTIVE_THRESHOLD]

    return {
        "avg_rating":       avg_rating,
        "avg_polarity":     avg_pol,
        "n_with_polarity":  len(with_pol),
        "pos_count":        len(pos),
        "neg_count":        len(neg),
        "neutral_count":    len(with_pol) - len(pos) - len(neg),
        "obj_neg_count":    len(obj_neg),
        "subj_neg_count":   len(subj_neg),
        "pos_pct":          len(pos) / n,
        "neg_pct":          len(neg) / n,
        "obj_neg_pct":      len(obj_neg) / n,
        "subj_neg_pct":     len(subj_neg) / n,
    }


def _issues_data(issues, corpus_df, total_reviews, top_n=TOP_ISSUES_N):
    """Top N priority issues with all per-card data ready for rendering."""
    out = []
    for rank, issue in enumerate(issues[:top_n], 1):
        cluster = issue["reviews"]
        label_aspects = _distinctive_aspects(cluster, corpus_df, total_reviews, k=4)
        aspect_string = ", ".join(label_aspects) if label_aspects else "no clear aspects"
        llm_label = issue.get("llm_label")

        obj, subj = _subjectivity_split(cluster)
        denom = len(obj) + len(subj)
        obj_share = len(obj) / denom if denom else 0.0
        subj_share = len(subj) / denom if denom else 0.0

        obj_aspects = _distinctive_aspects(obj, corpus_df, total_reviews, k=4) if obj else []
        subj_aspects = _distinctive_aspects(subj, corpus_df, total_reviews, k=4) if subj else []

        emotions = [r.get("emotion") for r in cluster if r.get("emotion")]
        ec = Counter(emotions)
        emotion_breakdown = [
            {"label": e, "count": c, "share": c / len(emotions)}
            for e, c in ec.most_common()
        ]

        entity_counter = Counter()
        entity_display = {}
        for r in cluster:
            for ent in r.get("entities") or []:
                key = ent["text"].lower()
                entity_counter[key] += 1
                entity_display.setdefault(key, ent["text"])
        top_entities = [
            {"text": entity_display[k], "count": c}
            for k, c in entity_counter.most_common(5)
            if c >= 3
        ]

        samples = _representative_reviews(cluster, n=3, max_len=200)

        sparkline = None
        counts, earliest, latest = bucket_dates(
            [r.get("date") for r in cluster], n_buckets=12
        )
        if counts is not None:
            sparkline = {
                "buckets":  counts,
                "earliest": earliest,
                "latest":   latest,
                "ascii":    render_sparkline(counts),
                "peak":     max(counts),
            }

        version_counter = Counter(
            r.get("app_version") for r in cluster if r.get("app_version")
        )
        app_versions = [
            {"version": v, "count": c}
            for v, c in version_counter.most_common(5)
        ]

        out.append({
            "rank":           rank,
            "cluster_id":     issue["cluster_id"],
            "label":          llm_label or aspect_string,
            "aspect_string":  aspect_string,
            "label_is_llm":   bool(llm_label),
            "count":          issue["count"],
            "avg_rating":     issue["avg_rating"],
            "avg_urgency":    issue["avg_urgency"],
            "avg_polarity":   issue["avg_polarity"],
            "score":          issue["score"],
            "components":     issue["components"],
            "bug_complaint": {
                "obj_share":    obj_share,
                "subj_share":   subj_share,
                "obj_aspects":  obj_aspects,
                "subj_aspects": subj_aspects,
                "obj_count":    len(obj),
                "subj_count":   len(subj),
            },
            "emotions":              emotion_breakdown,
            "entities":              top_entities,
            "representative_reviews": samples,
            "sparkline":             sparkline,
            "app_versions":          app_versions,
        })
    return out


def _run_delta_data(matches, resolved, issues, prior_run_id, app_slug):
    """Bucketed delta entries ready for rendering (or empty when not applicable)."""
    if not app_slug:
        return {"omitted": True}
    if not prior_run_id:
        return {"omitted": False, "first_run": True, "prior_run_id": None}

    cid_to_rank = {issue["cluster_id"]: i for i, issue in enumerate(issues, 1)}

    escalating, improving, new_, resolved_out = [], [], [], []
    for m in matches:
        kind = classify_delta(m)
        cur = m["current"]
        prior = m["prior"]
        label = _snap_label(cur, cid_to_rank)
        entry_base = {
            "label":         label,
            "cluster_id":    cur["cluster_id"],
            "current_count": cur["review_count"],
        }
        if kind == "escalating":
            pct = (cur["review_count"] / prior["review_count"] - 1) * 100
            escalating.append({
                **entry_base,
                "prior_count": prior["review_count"],
                "pct_change":  pct,
            })
        elif kind == "improving":
            pct = (1 - cur["review_count"] / prior["review_count"]) * 100
            improving.append({
                **entry_base,
                "prior_count": prior["review_count"],
                "pct_change":  -pct,
            })
        elif kind == "new" and cur["review_count"] >= DELTA_MIN_CLUSTER_SIZE:
            new_.append({**entry_base, "prior_count": None, "pct_change": None})

    for p in resolved:
        if p["review_count"] >= DELTA_MIN_CLUSTER_SIZE:
            resolved_out.append({
                "label":        _snap_label(p, {}),
                "cluster_id":   p["cluster_id"],
                "prior_count":  p["review_count"],
            })

    escalating.sort(key=lambda x: -x["pct_change"])
    improving.sort(key=lambda x: x["pct_change"])
    new_.sort(key=lambda x: -x["current_count"])
    resolved_out.sort(key=lambda x: -x["prior_count"])

    return {
        "omitted":      False,
        "first_run":    False,
        "prior_run_id": prior_run_id,
        "escalating":   escalating[:DELTA_PER_BUCKET],
        "improving":    improving[:DELTA_PER_BUCKET],
        "new":          new_[:DELTA_PER_BUCKET],
        "resolved":     resolved_out[:DELTA_PER_BUCKET],
    }


def _positives_data(reviews, corpus_df, top_n=TOP_POSITIVES_N):
    groups = _cluster_groups(reviews)
    if not groups:
        return {"items": []}
    total = len(reviews)
    positives = []
    for cid, group in groups.items():
        ratings = [r["rating"] for r in group if r.get("rating") is not None]
        if not ratings:
            continue
        avg_rating = float(np.mean(ratings))
        if avg_rating < 4.0:
            continue
        positives.append((cid, group, avg_rating))
    positives.sort(key=lambda x: -len(x[1]))
    entries = []
    for cid, group, avg_rating in positives[:top_n]:
        aspects = _distinctive_aspects(group, corpus_df, total, k=3)
        entries.append({
            "cluster_id":   cid,
            "label":        ", ".join(aspects) if aspects else "—",
            "count":        len(group),
            "avg_rating":   avg_rating,
        })
    return {"entries": entries}


def _absa_data(reviews, top_n=8):
    """ABSA loved/hated aspect rankings. Pure data — no markdown."""
    aspect_polarities = defaultdict(list)
    for r in reviews:
        for a in (r.get("aspects") or []):
            if isinstance(a, dict) and a.get("aspect"):
                aspect_polarities[a["aspect"]].append(a["polarity"])

    scored = {}
    for aspect, polarities in aspect_polarities.items():
        if len(polarities) < ABSA_MIN_MENTIONS:
            continue
        avg = float(np.mean(polarities))
        weight = math.log1p(len(polarities))
        scored[aspect] = (avg, len(polarities), avg * weight)

    loved = sorted(
        [(a, v) for a, v in scored.items() if v[0] > 0],
        key=lambda x: -x[1][2],
    )[:top_n]
    hated = sorted(
        [(a, v) for a, v in scored.items() if v[0] < 0],
        key=lambda x: x[1][2],
    )[:top_n]

    return {
        "loved": [{"aspect": a, "avg_polarity": v[0], "count": v[1]} for a, v in loved],
        "hated": [{"aspect": a, "avg_polarity": v[0], "count": v[1]} for a, v in hated],
    }


def _urgent_data(reviews, top_n=10, max_len=140):
    scored = [r for r in reviews if r.get("urgency") is not None and r.get("body")]
    scored.sort(key=lambda r: -r["urgency"])
    out = []
    for r in scored[:top_n]:
        body = r["body"][:max_len].replace("\n", " ").strip()
        out.append({
            "urgency": r["urgency"],
            "rating":  r.get("rating"),
            "emotion": r.get("emotion"),
            "body":    body,
        })
    return out


def _emotion_data(reviews):
    emotions = [r.get("emotion") for r in reviews if r.get("emotion")]
    counter = Counter(emotions)
    total = len(emotions)
    return {
        "entries": [
            {"emotion": e, "count": c, "share": c / total}
            for e, c in counter.most_common()
        ],
        "total": total,
    }


def _entities_data(reviews, top_n=10, min_count=3):
    counter = Counter()
    display = {}
    for r in reviews:
        for ent in r.get("entities") or []:
            key = ent["text"].lower()
            counter[key] += 1
            display.setdefault(key, ent["text"])
    return [
        {"text": display[k], "count": c}
        for k, c in counter.most_common(top_n)
        if c >= min_count
    ]


def _aspect_index_data(reviews, issues):
    if not issues:
        return []

    aspect_cluster_counts = defaultdict(Counter)
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is None:
            continue
        for a in set(_aspect_names(r)):
            aspect_cluster_counts[a][cid] += 1

    neg_aspect_count = Counter()
    for r in reviews:
        if r.get("rating") in (1, 2):
            for a in _aspect_names(r):
                neg_aspect_count[a] += 1

    top_aspects = [a for a, c in neg_aspect_count.most_common(15) if c >= 5]
    cid_to_rank = {issue["cluster_id"]: i for i, issue in enumerate(issues, 1)}

    out = []
    for aspect in top_aspects:
        cluster_counts = aspect_cluster_counts.get(aspect, Counter())
        top_cid = cluster_counts.most_common(1)[0][0] if cluster_counts else None
        rank = cid_to_rank.get(top_cid) if top_cid is not None else None
        out.append({
            "aspect":              aspect,
            "neg_count":           neg_aspect_count[aspect],
            "primary_issue_rank":  rank,
            "primary_cluster_id":  top_cid if rank is None else None,
        })
    return out


def _feature_summary_data(reviews, issues):
    return {
        "n_reviews":       len(reviews),
        "unique_aspects":  len({a for r in reviews for a in _aspect_names(r)}),
        "themes":          len({r.get("theme_cluster") for r in reviews
                                if r.get("theme_cluster") is not None}),
        "n_issues":        len(issues),
    }


def _run_summary_data(data):
    """
    Build the per-run narrative + the at-a-glance ribbon that lead the report.

    Two outputs:
        ribbon  — {reviews, escalating, new, resolved}; the four count fields
                  are ints when comparison is possible, None when it isn't
                  (first run for this app, or no app_slug supplied).
                  Renderers should display None as "—".
        narrative — a 1-2 sentence string describing this run in plain English.
                    Adapts to: no issues / first run / no prior / has prior.

    Reads from already-built sections of `data` (issues, run_delta,
    feature_summary, header) so it has to be computed after them.
    """
    issues = data["issues"]
    rd = data["run_delta"]
    n_reviews = data["header"]["review_count"]
    n_issues = data["feature_summary"]["n_issues"]

    # ----- ribbon counts ----------------------------------------------------
    has_comparison = not (rd.get("omitted") or rd.get("first_run"))
    if has_comparison:
        ribbon = {
            "reviews":    n_reviews,
            "escalating": len(rd.get("escalating", [])),
            "new":        len(rd.get("new", [])),
            "resolved":   len(rd.get("resolved", [])),
        }
    else:
        # First run / snapshot persistence off → no comparison data exists.
        ribbon = {
            "reviews":    n_reviews,
            "escalating": None,
            "new":        None,
            "resolved":   None,
        }

    # ----- narrative --------------------------------------------------------
    # Case A: no priority issues at all (very positive corpus or tiny dataset).
    if not issues:
        narrative = (
            f"{n_reviews:,} reviews analyzed — no negative-leaning clusters "
            f"detected this run."
        )
        return {"ribbon": ribbon, "narrative": narrative}

    top = issues[0]
    # Plain text — no markdown emphasis. Renderers can add their own styling
    # if they want (the HTML template wraps the label in <strong> via CSS,
    # markdown viewers show it as plain text).
    top_part = (
        f"Top issue: {top['label']} — {top['count']:,} reviews, "
        f"avg {top['avg_rating']:.1f}★, priority {top['score']:.2f}."
    )

    # Case B: first run for this app — no prior to compare against.
    if rd.get("first_run"):
        s = "s" if n_issues != 1 else ""
        change_part = (
            f"Baseline run — {n_issues} priority issue{s} identified. "
            f"Future runs will compare against this snapshot."
        )
    # Case C: snapshotting disabled (no app_slug) — no comparison either.
    elif rd.get("omitted"):
        s = "s" if n_issues != 1 else ""
        change_part = f"{n_issues} priority issue{s} this run."
    # Case D: has prior — describe what changed.
    else:
        esc = len(rd.get("escalating", []))
        imp = len(rd.get("improving", []))
        new_ = len(rd.get("new", []))
        res = len(rd.get("resolved", []))

        parts = []
        if esc:
            parts.append(f"{esc} escalating")
        if imp:
            parts.append(f"{imp} improving")
        if new_:
            parts.append(f"{new_} new")
        if res:
            parts.append(f"{res} resolved")

        if parts:
            change_part = ", ".join(parts) + " since prior run."
            # Capitalize first word for sentence-correctness
            change_part = change_part[0].upper() + change_part[1:]
        else:
            change_part = "No significant changes since prior run."

    narrative = f"{top_part} {change_part}"
    return {"ribbon": ribbon, "narrative": narrative}


# ---------------------------------------------------------------------------
# Markdown rendering (Phase VI: thin wrappers over the data dict)
# ---------------------------------------------------------------------------

def generate_report(reviews, app_name, app_slug=None, write_file=True):
    """Build the markdown report (and, in Phase VI, an HTML companion) from a
    list of feature-engineered reviews. Write both to disk and print a short
    terminal summary.

    When `app_slug` is supplied the run's per-cluster snapshots are persisted
    to issue_snapshots and the Run Delta section compares against the prior
    run for that slug. Pass `app_slug=None` to skip snapshotting (useful for
    ad-hoc / test invocations).
    """
    data = build_report_data(reviews, app_name, app_slug=app_slug)
    md = render_markdown(data)

    if write_file:
        os.makedirs(REPORT_DIR, exist_ok=True)
        run_id = data["_meta"]["run_id"]
        md_path = os.path.join(REPORT_DIR, f"{app_name}_{run_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote report to {md_path}")

        # HTML companion (Phase VI). Skipped silently if the renderer hasn't
        # been wired up or fails — markdown is the primary artifact.
        try:
            html = render_html(data)
            if html:
                html_path = os.path.join(REPORT_DIR, f"{app_name}_{run_id}.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"Wrote HTML report to {html_path}")
        except Exception as e:
            print(f"HTML render skipped: {e}")

    _print_terminal_summary(reviews, app_name, data["_meta"]["issues_full"])
    return md


_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_ASSET_DIR = os.path.join(_TEMPLATE_DIR, "assets")


def render_html(data):
    """Render the HTML report from a build_report_data() dict.

    Self-contained output: Pico CSS and Chart.js are inlined from
    pipeline/templates/assets/ so the resulting HTML works offline and can
    be shared as a single file. Falls back to None (which generate_report
    treats as "skip") if Jinja2 isn't installed or the template / assets are
    missing — markdown is the primary artifact, HTML is a polish-layer.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        return None

    pico_path = os.path.join(_ASSET_DIR, "pico.min.css")
    chart_path = os.path.join(_ASSET_DIR, "chart.umd.min.js")
    if not (os.path.exists(pico_path) and os.path.exists(chart_path)):
        return None

    with open(pico_path, "r", encoding="utf-8") as f:
        pico_css = f.read()
    with open(chart_path, "r", encoding="utf-8") as f:
        chart_js = f.read()

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.get_template("report.html.j2")

    # The JS only needs per-issue sparkline data — strip the rest to keep
    # the embedded payload small and avoid serializing non-JSON-safe values
    # (numpy arrays, embeddings) that live elsewhere in the data dict.
    js_data = {
        "issues": [
            {"sparkline": issue.get("sparkline")}
            for issue in data["issues"]
        ],
    }

    return template.render(
        elevator_pitch=ELEVATOR_PITCH,
        header=data["header"],
        run_summary=data["run_summary"],
        overall=data["overall"],
        issues=data["issues"],
        run_delta=data["run_delta"],
        positives=data["positives"],
        absa=data["absa"],
        urgent=data["urgent"],
        emotions=data["emotions"],
        entities=data["entities"],
        aspect_index=data["aspect_index"],
        feature_summary=data["feature_summary"],
        pico_css=pico_css,
        chart_js=chart_js,
        report_data=js_data,
    )


def render_markdown(data):
    """Render the full markdown report from a build_report_data() dict.

    Section order follows the editorial pass (Phase VII): pitch + auto-summary
    + ribbon lead, then Run Delta (the answer to 'what changed?'), then top
    priority issues, then a separator, then the supporting analysis sections.
    """
    sections = [
        _header_md(data["header"]),
        _run_summary_md(data["run_summary"]),
        _run_delta_md(data["run_delta"]),
        _priority_issues_md(data["issues"]),
        # Supporting analysis (lower visual weight in the HTML; in markdown
        # we keep them as ordinary sections but visually after a separator).
        _md_separator("Detailed analysis"),
        _overall_sentiment_md(data["overall"]),
        _positives_md(data["positives"]),
        _absa_md(data["absa"]),
        _urgent_md(data["urgent"]),
        _emotion_md(data["emotions"]),
        _entities_md(data["entities"]),
        _aspect_index_md(data["aspect_index"]),
        _feature_summary_md(data["feature_summary"]),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def _md_separator(label):
    """A horizontal rule + small heading marking the start of supporting
    detail. Keeps the markdown report scannable without making the
    sub-sections feel demoted in plain text the way they're demoted in HTML.
    """
    return f"---\n\n## {label}"


# ---------------------------------------------------------------------------
# Header + overall sentiment
# ---------------------------------------------------------------------------

def _header_md(data):
    return (
        f"# Review Analysis Report — {data['app_name']}\n"
        f"*{ELEVATOR_PITCH}*\n\n"
        f"{data['review_count']:,} reviews · generated {data['generated_at']} "
        f"· run `{data['run_id']}`"
    )


def _run_summary_md(data):
    """Render the at-a-glance ribbon + auto-narrative as a leading blockquote.

    Ribbon counts that aren't computable for this run (first run, no app_slug)
    render as em-dash so the field stays visible — the absence is informative.
    """
    ribbon = data["ribbon"]
    if ribbon["escalating"] is None:
        ribbon_line = (
            f"**{ribbon['reviews']:,}** reviews · — escalating · — new · — resolved"
        )
    else:
        ribbon_line = (
            f"**{ribbon['reviews']:,}** reviews · "
            f"**{ribbon['escalating']}** escalating · "
            f"**{ribbon['new']}** new · "
            f"**{ribbon['resolved']}** resolved"
        )
    # Blockquote so it stands out at the top in any markdown viewer.
    return f"> {data['narrative']}\n>\n> {ribbon_line}"


def _header(reviews, app_name, run_id):
    """Back-compat wrapper. New code should use _header_md(data)."""
    return _header_md(_header_data(reviews, app_name, run_id))


def _overall_sentiment_md(data):
    rows = [
        ("Avg rating", f"{data['avg_rating']:.2f} / 5"),
        ("Avg polarity", f"{data['avg_polarity']:.2f}"),
        ("% positive (polarity > 0.1)", f"{data['pos_pct']:.0%}"),
        ("% negative (polarity < -0.1)", f"{data['neg_pct']:.0%}"),
        ("% objective negative (bugs/issues)", f"{data['obj_neg_pct']:.0%}"),
        ("% subjective negative (emotional)", f"{data['subj_neg_pct']:.0%}"),
    ]
    body = "| Metric | Value |\n|---|---|\n"
    body += "\n".join(f"| {k} | {v} |" for k, v in rows)
    return "## Overall Sentiment\n" + body


def _overall_sentiment(reviews):
    """Back-compat wrapper."""
    return _overall_sentiment_md(_overall_sentiment_data(reviews))


# ---------------------------------------------------------------------------
# Issue scoring
# ---------------------------------------------------------------------------

def _cluster_groups(reviews):
    groups = defaultdict(list)
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is not None:
            groups[cid].append(r)
    return groups


def _score_issues(reviews):
    """
    Return a list of issue dicts (one per negative-leaning cluster), sorted by
    composite priority score (desc). Each entry includes the cluster id, its
    reviews, the score, and the four normalized component scores.
    """
    groups = _cluster_groups(reviews)
    if not groups:
        return []

    max_count = max(len(g) for g in groups.values())

    scored = []
    for cid, group in groups.items():
        ratings = [r["rating"] for r in group if r.get("rating") is not None]
        polarities = [r["polarity"] for r in group if r.get("polarity") is not None]
        avg_rating = float(np.mean(ratings)) if ratings else 0.0
        avg_pol = float(np.mean(polarities)) if polarities else 0.0

        is_issue = (
            (ratings and avg_rating < ISSUE_RATING_CEILING)
            or (polarities and avg_pol < ISSUE_POLARITY_CEILING)
        )
        if not is_issue:
            continue

        urgencies = [r["urgency"] for r in group if r.get("urgency") is not None]
        avg_urgency = float(np.mean(urgencies)) if urgencies else 0.0

        emotions = [r.get("emotion") for r in group if r.get("emotion")]
        intense = sum(1 for e in emotions if e in INTENSE_EMOTIONS)
        emotion_intensity = intense / len(emotions) if emotions else 0.0

        volume_score = math.log1p(len(group)) / math.log1p(max_count) if max_count else 0.0
        if ratings:
            severity_score = max(0.0, min(1.0, 1.0 - (avg_rating - 1.0) / 4.0))
        else:
            severity_score = 0.5

        priority = (
            W_VOLUME * volume_score
            + W_SEVERITY * severity_score
            + W_URGENCY * avg_urgency
            + W_EMOTION * emotion_intensity
        )

        scored.append({
            "cluster_id": cid,
            "reviews": group,
            "score": round(priority, 3),
            "count": len(group),
            "avg_rating": avg_rating,
            "avg_polarity": avg_pol,
            "avg_urgency": avg_urgency,
            "emotion_intensity": emotion_intensity,
            "components": {
                "volume": round(volume_score, 3),
                "severity": round(severity_score, 3),
                "urgency": round(avg_urgency, 3),
                "emotion": round(emotion_intensity, 3),
            },
        })

    scored.sort(key=lambda x: -x["score"])
    return scored


def _attach_llm_labels(issues, corpus_df, total_reviews, top_k_aspects=8):
    """
    Generate an LLM-written 4-6 word title for each priority issue and stash
    it on the issue dict as `llm_label`. Non-issue clusters aren't labeled
    (D1 — they don't appear in the issue cards or top of Run Delta, and
    skipping them halves API calls per run).

    On any failure (missing API key, network error, malformed response)
    `llm_label` is set to None and the renderer falls back to the
    aspect-string label.
    """
    for issue in issues:
        cluster = issue["reviews"]
        aspects = _distinctive_aspects(cluster, corpus_df, total_reviews, k=top_k_aspects)
        if not aspects:
            issue["llm_label"] = None
            continue
        samples = _representative_reviews(cluster, n=3, max_len=200)
        review_hashes = [compute_cache_key(r) for r in cluster]
        issue["llm_label"] = generate_cluster_label(aspects, samples, review_hashes)


def _build_snapshots(reviews, issues, corpus_df, total_reviews, top_k_aspects=8):
    """
    Build one snapshot dict per cluster (issues *and* non-issues) for
    persistence and cross-run matching. The snapshot captures everything
    needed to (a) match this cluster across future runs and (b) reconstruct
    a count time-series for it.

    Centroid is the mean of cluster review embeddings, in float32 to match
    the embedder's native precision and the DB blob format.
    """
    issue_cluster_ids = {i["cluster_id"] for i in issues}
    issue_score_by_cid = {i["cluster_id"]: i["score"] for i in issues}
    issue_llm_label_by_cid = {i["cluster_id"]: i.get("llm_label") for i in issues}

    groups = _cluster_groups(reviews)
    if not groups:
        return []

    snapshots = []
    for cid in sorted(groups.keys()):
        cluster = groups[cid]
        ratings = [r["rating"] for r in cluster if r.get("rating") is not None]
        polarities = [r["polarity"] for r in cluster if r.get("polarity") is not None]
        urgencies = [r["urgency"] for r in cluster if r.get("urgency") is not None]
        embeddings = [r["embedding"] for r in cluster if r.get("embedding") is not None]

        aspects = _distinctive_aspects(cluster, corpus_df, total_reviews, k=top_k_aspects)
        # Prefer the LLM label for priority issues; fall back to aspect string
        # for non-issue clusters and for any priority issue whose LLM call
        # failed (the snapshot is still saved either way).
        label = issue_llm_label_by_cid.get(cid) or (
            ", ".join(aspects[:5]) if aspects else None
        )

        centroid = None
        if embeddings:
            centroid = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0).tolist()

        snapshots.append({
            "cluster_id":     cid,
            "cluster_label":  label,
            "aspect_set":     aspects,
            "centroid":       centroid,
            "review_count":   len(cluster),
            "avg_rating":     float(np.mean(ratings)) if ratings else None,
            "avg_polarity":   float(np.mean(polarities)) if polarities else None,
            "avg_urgency":    float(np.mean(urgencies)) if urgencies else None,
            "priority_score": issue_score_by_cid.get(cid),
            "is_issue":       cid in issue_cluster_ids,
        })
    return snapshots


# ---------------------------------------------------------------------------
# Aspect ranking (TF-IDF style distinctiveness)
# ---------------------------------------------------------------------------

def _aspect_doc_freq(reviews):
    df = Counter()
    for r in reviews:
        for a in set(_aspect_names(r)):
            df[a] += 1
    return df


def _distinctive_aspects(cluster_reviews, corpus_df, total_reviews, k=4, min_count=3):
    """
    Return the k aspects most distinctive to this cluster, scored by
    cluster_count * log(total_reviews / corpus_df). Falls back to raw frequency
    on small clusters where the IDF term is unstable.
    """
    cluster_count = Counter()
    for r in cluster_reviews:
        for a in set(_aspect_names(r)):
            cluster_count[a] += 1

    if not cluster_count:
        return []

    if len(cluster_reviews) < TFIDF_MIN_CLUSTER_SIZE:
        return [a for a, _ in cluster_count.most_common(k)]

    scored = []
    for aspect, count in cluster_count.items():
        if count < min_count:
            continue
        df = corpus_df.get(aspect, 1)
        idf = math.log(total_reviews / df) if df else 0.0
        # 0 IDF means the aspect appears in every review of the corpus, so
        # it can't distinguish this cluster from any other. Drop it. Matches
        # the same filter in feature_engineering._distinctive_aspects.
        if idf <= 0:
            continue
        scored.append((aspect, count * idf))
    scored.sort(key=lambda x: -x[1])
    distinctive = [a for a, _ in scored[:k]]
    return distinctive or [a for a, _ in cluster_count.most_common(k)]


def _subjectivity_split(cluster_reviews):
    """Partition reviews into objective (bug-like) and subjective (complaint-like)."""
    objective, subjective = [], []
    for r in cluster_reviews:
        s = r.get("subjectivity")
        if s is None:
            continue
        if s < SUBJECTIVE_THRESHOLD:
            objective.append(r)
        else:
            subjective.append(r)
    return objective, subjective


# ---------------------------------------------------------------------------
# Priority Issues section (leaderboard + cards)
# ---------------------------------------------------------------------------

def _priority_issues_md(issues_data):
    if not issues_data:
        return "## Priority Issues\n_No negative-leaning clusters detected._"

    lines = [
        "## Priority Issues",
        "Negative-leaning clusters ranked by composite priority — "
        "weighted by volume, severity (avg rating), urgency, and "
        "intense-emotion share (anger / disgust / fear).",
        "",
        "| # | Issue | Reviews | Avg rating | Bug / Complaint | Priority |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for issue in issues_data:
        bc = issue["bug_complaint"]
        if bc["obj_count"] + bc["subj_count"]:
            split = f"{bc['obj_share']:.0%} / {bc['subj_share']:.0%}"
        else:
            split = "—"
        lines.append(
            f"| {issue['rank']} | {issue['label']} | {issue['count']} | "
            f"{issue['avg_rating']:.1f} | {split} | {issue['score']:.2f} |"
        )
    lines.append("")

    for issue in issues_data:
        lines.append(_issue_card_md(issue))
        lines.append("")

    return "\n".join(lines).rstrip()


def _issue_card_md(issue):
    bc = issue["bug_complaint"]
    emotion_str = (
        ", ".join(f"{e['label']} {e['share']:.0%}" for e in issue["emotions"][:3])
        if issue["emotions"] else "—"
    )
    entity_str = (
        ", ".join(f"{e['text']} ({e['count']})" for e in issue["entities"])
        if issue["entities"] else "—"
    )

    spark_line = ""
    if issue["sparkline"]:
        s = issue["sparkline"]
        spark_line = (
            f"**Trend ({s['earliest']} → {s['latest']}):** `{s['ascii']}` "
            f"({len(s['buckets'])} buckets, peak {s['peak']} mentions)"
        )

    version_line = ""
    if issue["app_versions"]:
        version_line = "**By app version:** " + ", ".join(
            f"{v['version']} ({v['count']})" for v in issue["app_versions"]
        )

    lines = [
        f"### Issue {issue['rank']} — {issue['label']}  (priority {issue['score']:.2f})",
    ]
    if issue["label_is_llm"]:
        lines.append(f"*{issue['aspect_string']}*")
    lines += [
        f"**{issue['count']:,} reviews** · avg rating {issue['avg_rating']:.1f} "
        f"· avg urgency {issue['avg_urgency']:.2f}",
        "",
        "| Side | Share | Top aspects |",
        "|---|---:|---|",
        f"| Objective (bugs) | {bc['obj_share']:.0%} | "
        f"{', '.join(bc['obj_aspects']) or '—'} |",
        f"| Subjective (complaints) | {bc['subj_share']:.0%} | "
        f"{', '.join(bc['subj_aspects']) or '—'} |",
        "",
        f"**Top emotions:** {emotion_str}",
        f"**Mentioned entities:** {entity_str}",
    ]
    if spark_line:
        lines += ["", spark_line]
    if version_line:
        lines += ["", version_line]
    lines += ["", "Representative reviews:"]
    if issue["representative_reviews"]:
        for s in issue["representative_reviews"]:
            # Markdown bullet — the data layer keeps reviews at length 200,
            # truncate to MD-friendly 140 here for compactness.
            lines.append(f"- {s[:140]}")
    else:
        lines.append("- _No representative reviews available._")
    return "\n".join(lines)


def _priority_issues_section(reviews, issues, corpus_df):
    """Back-compat wrapper."""
    total = len(reviews)
    return _priority_issues_md(_issues_data(issues, corpus_df, total))


# ---------------------------------------------------------------------------
# Run-to-run delta (Phase IV)
# ---------------------------------------------------------------------------

# Below this size we don't surface a cluster in the new/resolved buckets —
# tiny clusters churn between runs as KMeans labels shuffle and would just
# create noise.
DELTA_MIN_CLUSTER_SIZE = 20
DELTA_PER_BUCKET = 5  # cap entries shown per bucket


def _snap_label(snap, cid_to_rank):
    """Render a snapshot as 'Issue N (Slow delivery and damaged packaging)'
    if it's a current priority issue, otherwise '(...)'. Prefers the
    snapshot's stored cluster_label (LLM-written for issue clusters,
    aspect-string for non-issue clusters); falls back to the raw aspect set.
    """
    label = snap.get("cluster_label")
    if not label:
        aspects = snap.get("aspect_set") or []
        label = ", ".join(aspects[:3]) if aspects else f"cluster {snap['cluster_id']}"
    rank = cid_to_rank.get(snap["cluster_id"])
    if rank:
        return f"Issue {rank} ({label})"
    return f"({label})"


def _run_delta_md(data):
    """Render the Run Delta section from the data dict."""
    if data.get("omitted"):
        return ""
    if data.get("first_run"):
        return (
            "## Run Delta\n"
            "_No prior run on file for this app — this is the baseline. "
            "The next run will compare against this snapshot._"
        )

    prior_run_id = data["prior_run_id"]
    escalating = data["escalating"]
    improving = data["improving"]
    new_ = data["new"]
    resolved = data["resolved"]

    if not (escalating or improving or new_ or resolved):
        return (
            f"## Run Delta\n"
            f"Compared to prior run `{prior_run_id}`. "
            f"_No significant changes since prior run._"
        )

    lines = [
        "## Run Delta",
        f"Compared to prior run `{prior_run_id}`. Issues that grew, shrank, "
        "appeared, or disappeared since then.",
    ]
    if escalating:
        lines += ["", "### Escalating (>20% increase in mentions)"] + [
            f"- {e['label']} — {e['prior_count']:,} → {e['current_count']:,} "
            f"reviews (+{e['pct_change']:.0f}%)"
            for e in escalating
        ]
    if improving:
        lines += ["", "### Improving (>20% decrease in mentions)"] + [
            f"- {e['label']} — {e['prior_count']:,} → {e['current_count']:,} "
            f"reviews ({e['pct_change']:.0f}%)"
            for e in improving
        ]
    if new_:
        lines += ["", "### New (no match in prior run)"] + [
            f"- {e['label']} — {e['current_count']:,} reviews this run, no prior match"
            for e in new_
        ]
    if resolved:
        lines += ["", "### Resolved (prior issue with no current match)"] + [
            f"- {e['label']} — was {e['prior_count']:,} reviews last run, no current match"
            for e in resolved
        ]
    return "\n".join(lines)


def _run_delta_section(matches, resolved, issues, prior_run_id, app_slug):
    """Back-compat wrapper. New code should consume the data dict directly."""
    return _run_delta_md(_run_delta_data(matches, resolved, issues, prior_run_id, app_slug))


# ---------------------------------------------------------------------------
# Positives, urgency, emotion, entities, aspect index
# ---------------------------------------------------------------------------

def _positives_md(data):
    entries = data.get("entries") or []
    if not entries:
        return "## Top Positives\n_No strongly positive clusters detected._"

    lines = [
        "## Top Positives",
        "Clusters where users express satisfaction — useful for "
        "marketing and identifying what to preserve.",
        "",
        "| Cluster | Top aspects | Reviews | Avg rating |",
        "|---|---|---:|---:|",
    ]
    for it in entries:
        lines.append(
            f"| {it['cluster_id']} | {it['label']} | {it['count']} | "
            f"{it['avg_rating']:.1f} |"
        )
    return "\n".join(lines)


def _top_positives_section(reviews, corpus_df):
    """Back-compat wrapper."""
    return _positives_md(_positives_data(reviews, corpus_df))


def _representative_reviews(cluster_reviews, n=3, max_len=140):
    """Pick `n` representative reviews from a cluster, deduped by body text.

    Two-stage selection (Phase V):
      1. Build a candidate pool of the reviews closest to the cluster centroid
         — at least 3n, or the top quarter of the cluster, whichever is larger.
         This guarantees every candidate is plausibly on-theme.
      2. Within the pool, rank by thumbs_up DESC, then by closeness ASC as a
         tiebreaker. Surfaces reviews other readers found helpful, while
         keeping out far-from-centroid outliers that just happen to be popular.

    On older CSVs that predate the cleaner fix every review has thumbs_up=0,
    in which case stage 2's tiebreaker (closeness) takes over and behavior
    matches the prior centroid-only selection.
    """
    valid = [r for r in cluster_reviews if r.get("embedding") and r.get("body")]
    if not valid:
        return []

    embeds = np.array([r["embedding"] for r in valid])
    centroid = embeds.mean(axis=0)
    distances = np.linalg.norm(embeds - centroid, axis=1)

    # Stage 1: top closest = candidate pool. Minimum n so we have enough to
    # pick from after dedup; otherwise the top quarter — small enough to
    # exclude outliers, large enough to give thumbs_up meaningful pull on
    # real-size clusters.
    pool_size = max(n, len(valid) // 4)
    pool = np.argsort(distances)[:pool_size]

    # Stage 2: within pool, prefer high-thumbs reviews; closer wins on ties.
    def rank_key(idx):
        thumbs = valid[idx].get("thumbs_up") or 0
        return (-int(thumbs), float(distances[idx]))

    ordered = sorted(pool, key=rank_key)

    picked = []
    seen = set()
    for idx in ordered:
        body = valid[idx]["body"].replace("\n", " ").strip()
        key = body.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(body[:max_len])
        if len(picked) == n:
            break
    return picked


def _emotion_md(data):
    entries = data["entries"]
    if not entries:
        return "## Emotion Distribution\n_No emotion data available._"
    body = "| Emotion | Count | Share |\n|---|---:|---:|\n"
    body += "\n".join(
        f"| {it['emotion']} | {it['count']} | {it['share']:.0%} |"
        for it in entries
    )
    return "## Emotion Distribution\n" + body


def _emotion_section(reviews):
    """Back-compat wrapper."""
    return _emotion_md(_emotion_data(reviews))


def _urgent_md(data):
    if not data:
        return "## Most Urgent Reviews\n_No urgency scores available._"
    lines = [
        "## Most Urgent Reviews (by actionability score)",
        "| Urgency | Rating | Emotion | Review |",
        "|---:|---:|---|---|",
    ]
    for r in data:
        body = r["body"].replace("|", "/")
        rating = r["rating"] if r["rating"] is not None else "?"
        emotion = r["emotion"] or "?"
        lines.append(f"| {r['urgency']:.2f} | {rating} | {emotion} | {body} |")
    return "\n".join(lines)


def _urgent_issues_section(reviews, top_n=10):
    """Back-compat wrapper."""
    return _urgent_md(_urgent_data(reviews, top_n=top_n))


def _entities_md(data):
    if not data:
        return "## Mentioned Entities (companies & products)\n_No recurring entities detected._"
    body = "| Entity | Mentions |\n|---|---:|\n"
    body += "\n".join(f"| {e['text']} | {e['count']} |" for e in data)
    return "## Mentioned Entities (companies & products)\n" + body


def _entities_section(reviews, top_n=10, min_count=3):
    """Back-compat wrapper."""
    return _entities_md(_entities_data(reviews, top_n=top_n, min_count=min_count))


def _aspect_index_md(data):
    if not data:
        return ""

    lines = [
        "## Aspect Index",
        "Top aspects from 1–2★ reviews and the priority issue they "
        "most belong to. Use this to look up which issue covers a "
        "specific feature complaint.",
        "",
        "| Aspect | Mentions in 1–2★ | Primary issue |",
        "|---|---:|---|",
    ]
    for entry in data:
        if entry["primary_issue_rank"]:
            primary = f"Issue {entry['primary_issue_rank']}"
        elif entry["primary_cluster_id"] is not None:
            primary = f"cluster {entry['primary_cluster_id']} (not in top issues)"
        else:
            primary = "—"
        lines.append(f"| {entry['aspect']} | {entry['neg_count']} | {primary} |")
    return "\n".join(lines)


def _aspect_index_section(reviews, issues):
    """Back-compat wrapper."""
    return _aspect_index_md(_aspect_index_data(reviews, issues))


def _absa_md(data):
    loved = data.get("loved") or []
    hated = data.get("hated") or []
    if not loved and not hated:
        return ""

    lines = [
        "## Aspect Sentiment (ABSA)",
        "Per-aspect polarity scored independently for each (review, aspect) pair "
        "by DeBERTa ABSA. Ranked by avg polarity × log(mentions) — balances "
        "confidence against volume.",
    ]

    if loved:
        lines += [
            "",
            "### Top Loved Features",
            "| Aspect | Avg polarity | Mentions |",
            "|---|---:|---:|",
        ]
        for it in loved:
            lines.append(f"| {it['aspect']} | +{it['avg_polarity']:.2f} | {it['count']} |")

    if hated:
        lines += [
            "",
            "### Top Hated Features",
            "| Aspect | Avg polarity | Mentions |",
            "|---|---:|---:|",
        ]
        for it in hated:
            lines.append(f"| {it['aspect']} | {it['avg_polarity']:.2f} | {it['count']} |")

    return "\n".join(lines)


def _absa_section(reviews, top_n=8):
    """Back-compat wrapper."""
    return _absa_md(_absa_data(reviews, top_n=top_n))


def _feature_summary_md(data):
    return (
        "## Feature Summary\n"
        f"- {data['n_reviews']:,} reviews processed\n"
        f"- {data['unique_aspects']:,} unique aspects extracted\n"
        f"- {data['themes']} themes discovered\n"
        f"- {data['n_issues']} negative-leaning issues identified"
    )


def _feature_summary(reviews, issues):
    """Back-compat wrapper."""
    return _feature_summary_md(_feature_summary_data(reviews, issues))


def _print_terminal_summary(reviews, app_name, issues):
    n = len(reviews)
    with_pol = [r for r in reviews if r.get("polarity") is not None]
    avg_pol = np.mean([r["polarity"] for r in with_pol]) if with_pol else 0.0
    themes = len({r.get("theme_cluster") for r in reviews if r.get("theme_cluster") is not None})

    print(f"\n--- Summary: {app_name} ---")
    print(f"Reviews:      {n:,}")
    print(f"Avg polarity: {avg_pol:.2f}")
    print(f"Themes:       {themes}")
    print(f"Issues:       {len(issues)}")
    if issues:
        top = issues[0]
        print(f"Top issue:    priority {top['score']:.2f} ({top['count']} reviews, avg rating {top['avg_rating']:.1f})")
