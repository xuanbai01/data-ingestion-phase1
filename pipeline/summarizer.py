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
from datetime import datetime
import numpy as np

REPORT_DIR = "reports"

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


def generate_report(reviews, app_name, write_file=True):
    """Build the full markdown report, write to file, and print a terminal summary."""
    issues = _score_issues(reviews)
    corpus_df = _aspect_doc_freq(reviews)

    sections = [
        _header(reviews, app_name),
        _overall_sentiment(reviews),
        _priority_issues_section(reviews, issues, corpus_df),
        _top_positives_section(reviews, corpus_df),
        _urgent_issues_section(reviews),
        _emotion_section(reviews),
        _entities_section(reviews),
        _aspect_index_section(reviews, issues),
        _feature_summary(reviews, issues),
    ]
    report = "\n\n".join(s for s in sections if s) + "\n"

    if write_file:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORT_DIR, f"{app_name}_{timestamp}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote report to {path}")

    _print_terminal_summary(reviews, app_name, issues)
    return report


# ---------------------------------------------------------------------------
# Header + overall sentiment
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Aspect ranking (TF-IDF style distinctiveness)
# ---------------------------------------------------------------------------

def _aspect_doc_freq(reviews):
    df = Counter()
    for r in reviews:
        for a in set(r.get("aspects") or []):
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
        for a in set(r.get("aspects") or []):
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

def _priority_issues_section(reviews, issues, corpus_df):
    if not issues:
        return "## Priority Issues\n_No negative-leaning clusters detected._"

    total = len(reviews)
    top = issues[:TOP_ISSUES_N]

    lines = [
        "## Priority Issues",
        "Negative-leaning clusters ranked by composite priority — "
        "weighted by volume, severity (avg rating), urgency, and "
        "intense-emotion share (anger / disgust / fear).",
        "",
        "| # | Issue | Reviews | Avg rating | Bug / Complaint | Priority |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for i, issue in enumerate(top, 1):
        cluster = issue["reviews"]
        label = ", ".join(_distinctive_aspects(cluster, corpus_df, total, k=3)) or "—"
        obj, subj = _subjectivity_split(cluster)
        denom = len(obj) + len(subj)
        if denom:
            obj_pct = len(obj) / denom
            subj_pct = len(subj) / denom
            split = f"{obj_pct:.0%} / {subj_pct:.0%}"
        else:
            split = "—"
        lines.append(
            f"| {i} | {label} | {issue['count']} | "
            f"{issue['avg_rating']:.1f} | {split} | {issue['score']:.2f} |"
        )
    lines.append("")

    for i, issue in enumerate(top, 1):
        lines.append(_issue_card(issue, i, corpus_df, total))
        lines.append("")

    return "\n".join(lines).rstrip()


def _issue_card(issue, rank, corpus_df, total_reviews):
    cluster = issue["reviews"]
    label_aspects = _distinctive_aspects(cluster, corpus_df, total_reviews, k=4)
    label = ", ".join(label_aspects) if label_aspects else "no clear aspects"

    obj, subj = _subjectivity_split(cluster)
    denom = len(obj) + len(subj)
    obj_share = len(obj) / denom if denom else 0.0
    subj_share = len(subj) / denom if denom else 0.0

    obj_aspects = _distinctive_aspects(obj, corpus_df, total_reviews, k=4) if obj else []
    subj_aspects = _distinctive_aspects(subj, corpus_df, total_reviews, k=4) if subj else []

    emotions = [r.get("emotion") for r in cluster if r.get("emotion")]
    if emotions:
        ec = Counter(emotions)
        emotion_str = ", ".join(
            f"{e} {c / len(emotions):.0%}" for e, c in ec.most_common(3)
        )
    else:
        emotion_str = "—"

    entity_counter = Counter()
    entity_display = {}
    for r in cluster:
        for ent in r.get("entities") or []:
            key = ent["text"].lower()
            entity_counter[key] += 1
            entity_display.setdefault(key, ent["text"])
    top_entities = [
        (entity_display[k], c)
        for k, c in entity_counter.most_common(5)
        if c >= 3
    ]
    entity_str = (
        ", ".join(f"{name} ({c})" for name, c in top_entities)
        if top_entities else "—"
    )

    samples = _representative_reviews(cluster, n=3)

    lines = [
        f"### Issue {rank} — {label}  (priority {issue['score']:.2f})",
        f"**{issue['count']:,} reviews** · avg rating {issue['avg_rating']:.1f} "
        f"· avg urgency {issue['avg_urgency']:.2f}",
        "",
        "| Side | Share | Top aspects |",
        "|---|---:|---|",
        f"| Objective (bugs) | {obj_share:.0%} | {', '.join(obj_aspects) or '—'} |",
        f"| Subjective (complaints) | {subj_share:.0%} | {', '.join(subj_aspects) or '—'} |",
        "",
        f"**Top emotions:** {emotion_str}",
        f"**Mentioned entities:** {entity_str}",
        "",
        "Representative reviews:",
    ]
    if samples:
        for s in samples:
            lines.append(f"- {s}")
    else:
        lines.append("- _No representative reviews available._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Positives, urgency, emotion, entities, aspect index
# ---------------------------------------------------------------------------

def _top_positives_section(reviews, corpus_df):
    """Smaller positives view — clusters that skew positive, with distinctive aspects."""
    groups = _cluster_groups(reviews)
    if not groups:
        return ""

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

    if not positives:
        return "## Top Positives\n_No strongly positive clusters detected._"

    positives.sort(key=lambda x: -len(x[1]))
    positives = positives[:TOP_POSITIVES_N]

    lines = [
        "## Top Positives",
        "Clusters where users express satisfaction — useful for "
        "marketing and identifying what to preserve.",
        "",
        "| Cluster | Top aspects | Reviews | Avg rating |",
        "|---|---|---:|---:|",
    ]
    for cid, group, avg_rating in positives:
        label = ", ".join(_distinctive_aspects(group, corpus_df, total, k=3)) or "—"
        lines.append(f"| {cid} | {label} | {len(group)} | {avg_rating:.1f} |")
    return "\n".join(lines)


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


def _aspect_index_section(reviews, issues):
    """Map top 1-2★ aspects to the priority issue they primarily belong to."""
    if not issues:
        return ""

    aspect_cluster_counts = defaultdict(Counter)
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is None:
            continue
        for a in set(r.get("aspects") or []):
            aspect_cluster_counts[a][cid] += 1

    neg_aspect_count = Counter()
    for r in reviews:
        if r.get("rating") in (1, 2):
            for a in r.get("aspects") or []:
                neg_aspect_count[a] += 1

    top_aspects = [a for a, c in neg_aspect_count.most_common(15) if c >= 5]
    if not top_aspects:
        return ""

    cid_to_rank = {issue["cluster_id"]: i for i, issue in enumerate(issues, 1)}

    lines = [
        "## Aspect Index",
        "Top aspects from 1–2★ reviews and the priority issue they "
        "most belong to. Use this to look up which issue covers a "
        "specific feature complaint.",
        "",
        "| Aspect | Mentions in 1–2★ | Primary issue |",
        "|---|---:|---|",
    ]
    for aspect in top_aspects:
        cluster_counts = aspect_cluster_counts.get(aspect, Counter())
        if not cluster_counts:
            primary = "—"
        else:
            top_cid, _ = cluster_counts.most_common(1)[0]
            rank = cid_to_rank.get(top_cid)
            primary = f"Issue {rank}" if rank else f"cluster {top_cid} (not in top issues)"
        lines.append(f"| {aspect} | {neg_aspect_count[aspect]} | {primary} |")
    return "\n".join(lines)


def _feature_summary(reviews, issues):
    n = len(reviews)
    unique_aspects = len({a for r in reviews for a in (r.get("aspects") or [])})
    themes = len({r.get("theme_cluster") for r in reviews if r.get("theme_cluster") is not None})
    return (
        "## Feature Summary\n"
        f"- {n:,} reviews processed\n"
        f"- {unique_aspects:,} unique aspects extracted\n"
        f"- {themes} themes discovered\n"
        f"- {len(issues)} negative-leaning issues identified"
    )


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
