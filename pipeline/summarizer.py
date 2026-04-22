"""
Summarizer: consumes feature-engineered reviews and produces a human-readable
markdown report plus a short terminal summary.

Each markdown section is built by its own small function so that new feature
columns (e.g. emotion, urgency) can add a section without touching the others.
"""
import os
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

REPORT_DIR = "reports"

NEG_THRESHOLD = -0.1
POS_THRESHOLD = 0.1
SUBJECTIVE_THRESHOLD = 0.5


def generate_report(reviews, app_name, write_file=True):
    """Build the full markdown report, write to file, and print a terminal summary."""
    sections = [
        _header(reviews, app_name),
        _overall_sentiment(reviews),
        _emotion_section(reviews),
        _top_aspects_tables(reviews),
        _urgent_issues_section(reviews),
        _entities_section(reviews),
        _themes_section(reviews),
        _feature_summary(reviews),
    ]
    report = "\n\n".join(sections) + "\n"

    if write_file:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORT_DIR, f"{app_name}_{timestamp}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote report to {path}")

    _print_terminal_summary(reviews, app_name)
    return report


def _header(reviews, app_name):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    return (
        f"# Review Analysis Report — {app_name}\n"
        f"*{len(reviews):,} reviews · generated {date}*"
    )


def _overall_sentiment(reviews):
    rated = [r for r in reviews if r.get("rating") is not None]
    with_pol = [r for r in reviews if r.get("polarity") is not None]
    n = len(with_pol) or 1

    avg_rating = np.mean([r["rating"] for r in rated]) if rated else 0.0
    avg_pol = np.mean([r["polarity"] for r in with_pol]) if with_pol else 0.0

    neg = [r for r in with_pol if r["polarity"] < NEG_THRESHOLD]
    pos = [r for r in with_pol if r["polarity"] > POS_THRESHOLD]
    obj_neg = [r for r in neg if (r.get("subjectivity") or 0) < SUBJECTIVE_THRESHOLD]
    subj_neg = [r for r in neg if (r.get("subjectivity") or 0) >= SUBJECTIVE_THRESHOLD]

    rows = [
        ("Avg rating", f"{avg_rating:.2f} / 5"),
        ("Avg polarity", f"{avg_pol:.2f}"),
        ("% positive (polarity > 0.1)", f"{len(pos) / n:.0%}"),
        ("% negative (polarity < -0.1)", f"{len(neg) / n:.0%}"),
        ("% objective negative (bugs/issues)", f"{len(obj_neg) / n:.0%}"),
        ("% subjective negative (emotional)", f"{len(subj_neg) / n:.0%}"),
    ]
    body = "| Metric | Value |\n|---|---|\n"
    body += "\n".join(f"| {k} | {v} |" for k, v in rows)
    return "## Overall Sentiment\n" + body


def _aspect_stats(reviews_subset):
    """Aggregate per-aspect: count, polarities, ratings."""
    stats = defaultdict(lambda: {"count": 0, "polarity": [], "rating": []})
    for r in reviews_subset:
        for a in r.get("aspects") or []:
            stats[a]["count"] += 1
            if r.get("polarity") is not None:
                stats[a]["polarity"].append(r["polarity"])
            if r.get("rating") is not None:
                stats[a]["rating"].append(r["rating"])
    return stats


def _format_aspect_table(reviews_subset, top_n=10, min_count=5):
    stats = _aspect_stats(reviews_subset)
    rows = []
    for aspect, s in stats.items():
        if s["count"] < min_count:
            continue
        avg_pol = np.mean(s["polarity"]) if s["polarity"] else 0.0
        avg_rating = np.mean(s["rating"]) if s["rating"] else 0.0
        rows.append((aspect, s["count"], avg_pol, avg_rating))
    rows.sort(key=lambda r: -r[1])
    rows = rows[:top_n]

    if not rows:
        return "_No aspects met the minimum count threshold._"

    body = "| Aspect | Mentions | Avg polarity | Avg rating |\n|---|---:|---:|---:|\n"
    body += "\n".join(
        f"| {a} | {c} | {p:.2f} | {r:.1f} |" for a, c, p, r in rows
    )
    return body


def _top_aspects_tables(reviews):
    neg_subset = [r for r in reviews if r.get("rating") in (1, 2)]
    pos_subset = [r for r in reviews if r.get("rating") in (4, 5)]

    return (
        "## Top Issues (1-2 star reviews)\n"
        + _format_aspect_table(neg_subset)
        + "\n\n## Top Positives (4-5 star reviews)\n"
        + _format_aspect_table(pos_subset)
    )


def _representative_reviews(cluster_reviews, n=3, max_len=140):
    """Pick the n reviews closest to the cluster centroid, deduped by body text."""
    valid = [r for r in cluster_reviews if r.get("embedding") and r.get("body")]
    if not valid:
        return []
    embeds = np.array([r["embedding"] for r in valid])
    centroid = embeds.mean(axis=0)
    distances = np.linalg.norm(embeds - centroid, axis=1)
    order = np.argsort(distances)

    picked = []
    seen = set()
    for idx in order:
        body = valid[idx]["body"].replace("\n", " ").strip()
        key = body.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(body[:max_len])
        if len(picked) == n:
            break
    return picked


def _themes_section(reviews):
    clusters = defaultdict(list)
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is not None:
            clusters[cid].append(r)

    lines = ["## Themes (auto-clustered)"]
    sorted_ids = sorted(clusters.keys(), key=lambda c: -len(clusters[c]))
    for cid in sorted_ids:
        group = clusters[cid]
        aspect_counter = Counter(
            a for r in group for a in (r.get("aspects") or [])
        )
        top_aspects = [a for a, _ in aspect_counter.most_common(4)]
        label = ", ".join(top_aspects) if top_aspects else "no aspects"

        ratings = [r["rating"] for r in group if r.get("rating") is not None]
        avg_rating = np.mean(ratings) if ratings else 0.0
        pols = [r["polarity"] for r in group if r.get("polarity") is not None]
        avg_pol = np.mean(pols) if pols else 0.0

        lines.append(
            f"### Theme {cid} — top aspects: {label} "
            f"({len(group)} reviews, avg rating {avg_rating:.1f}, avg polarity {avg_pol:.2f})"
        )
        lines.append("Representative reviews:")
        for sample in _representative_reviews(group):
            lines.append(f"- {sample}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _emotion_section(reviews):
    emotions = [r.get("emotion") for r in reviews if r.get("emotion")]
    if not emotions:
        return "## Emotion Distribution\n_No emotion data available._"

    total = len(emotions)
    counter = Counter(emotions)
    body = "| Emotion | Count | Share |\n|---|---:|---:|\n"
    body += "\n".join(
        f"| {e} | {c} | {c / total:.0%} |"
        for e, c in counter.most_common()
    )
    return "## Emotion Distribution\n" + body


def _urgent_issues_section(reviews, top_n=10):
    scored = [
        r for r in reviews
        if r.get("urgency") is not None and r.get("body")
    ]
    scored.sort(key=lambda r: -r["urgency"])
    top = scored[:top_n]
    if not top:
        return "## Most Urgent Reviews\n_No urgency scores available._"

    lines = [
        "## Most Urgent Reviews (by actionability score)",
        "| Urgency | Rating | Emotion | Review |",
        "|---:|---:|---|---|",
    ]
    for r in top:
        body = r["body"][:140].replace("\n", " ").replace("|", "/").strip()
        rating = r.get("rating") if r.get("rating") is not None else "?"
        emotion = r.get("emotion") or "?"
        lines.append(f"| {r['urgency']:.2f} | {rating} | {emotion} | {body} |")
    return "\n".join(lines)


def _entities_section(reviews, top_n=10, min_count=3):
    counter = Counter()
    display = {}
    for r in reviews:
        for ent in r.get("entities") or []:
            key = ent["text"].lower()
            counter[key] += 1
            display.setdefault(key, ent["text"])

    rows = [(display[k], c) for k, c in counter.most_common(top_n) if c >= min_count]
    if not rows:
        return "## Mentioned Entities (companies & products)\n_No recurring entities detected._"

    body = "| Entity | Mentions |\n|---|---:|\n"
    body += "\n".join(f"| {name} | {count} |" for name, count in rows)
    return "## Mentioned Entities (companies & products)\n" + body


def _feature_summary(reviews):
    n = len(reviews)
    unique_aspects = len({a for r in reviews for a in (r.get("aspects") or [])})
    themes = len({r.get("theme_cluster") for r in reviews if r.get("theme_cluster") is not None})
    return (
        "## Feature Summary\n"
        f"- {n:,} reviews processed\n"
        f"- {unique_aspects:,} unique aspects extracted\n"
        f"- {themes} themes discovered"
    )


def _print_terminal_summary(reviews, app_name):
    n = len(reviews)
    with_pol = [r for r in reviews if r.get("polarity") is not None]
    avg_pol = np.mean([r["polarity"] for r in with_pol]) if with_pol else 0.0
    themes = len({r.get("theme_cluster") for r in reviews if r.get("theme_cluster") is not None})

    print(f"\n--- Summary: {app_name} ---")
    print(f"Reviews:      {n:,}")
    print(f"Avg polarity: {avg_pol:.2f}")
    print(f"Themes:       {themes}")
