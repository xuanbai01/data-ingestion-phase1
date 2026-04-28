import math
import os
from collections import Counter, defaultdict

from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from transformers import pipeline as hf_pipeline
from tqdm import tqdm
import numpy as np

from database.db import (
    compute_cache_key,
    load_features_batch,
    save_features_batch,
)

# ---------------------------------------------------------------------------
# Device selection
#
# Auto-detect CUDA. If a GPU is available and FORCE_CPU is not set, all three
# models (embedder, emotion classifier, ABSA classifier) move to GPU.
# Override with FORCE_CPU=1 to fall back to CPU even on a GPU machine
# (useful for parity with CI / non-GPU teammates).
# ---------------------------------------------------------------------------
_force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
USE_GPU = torch.cuda.is_available() and not _force_cpu
HF_DEVICE = 0 if USE_GPU else -1            # transformers pipeline convention
ST_DEVICE = "cuda" if USE_GPU else "cpu"    # sentence-transformers convention
INFERENCE_BATCH = 64 if USE_GPU else 32      # larger batches pay off on GPU

# ---------------------------------------------------------------------------
# Versioning
#
# Bump FEATURE_SCHEMA_VERSION whenever any feature's semantics change
# (new model, new filter, new field). Rows in the cache at older versions
# are treated as misses and recomputed transparently.
#
# v2: swapped embedder all-MiniLM-L6-v2 (384d) → all-mpnet-base-v2 (768d)
#     for better cluster separation.
# v3: aspects: list[str] → list[{aspect, polarity, confidence}] via DeBERTa ABSA.
# ---------------------------------------------------------------------------
FEATURE_SCHEMA_VERSION = 3

EMBEDDER_MODEL_NAME = "all-mpnet-base-v2"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
SPACY_MODEL_NAME = "en_core_web_sm"

ABSA_MAX_ASPECTS = 8  # cap per review to bound inference time on very long aspect lists

# Clustering defaults
DEFAULT_K_RANGE = (4, 16)        # inclusive low, exclusive high
SILHOUETTE_SAMPLE_SIZE = 2000    # silhouette is O(n²); sample on large corpora
MERGE_TOP_K = 8                  # number of top aspects compared when merging
MERGE_THRESHOLD = 0.6            # overlap-coefficient threshold for merging

# Load models once at module level so they are not reloaded on every call.
# Device is auto-detected above; print on import so the run log records it.
print(f"[feature_engineering] device: {'cuda' if USE_GPU else 'cpu'}"
      f"{' (FORCE_CPU set)' if _force_cpu and torch.cuda.is_available() else ''}")
nlp = spacy.load(SPACY_MODEL_NAME)
embedding_model = SentenceTransformer(EMBEDDER_MODEL_NAME, device=ST_DEVICE)
emotion_classifier = hf_pipeline(
    "text-classification",
    model=EMOTION_MODEL_NAME,
    top_k=1,
    device=HF_DEVICE,
)
absa_classifier = hf_pipeline(
    "text-classification",
    model=ABSA_MODEL_NAME,
    top_k=1,
    device=HF_DEVICE,
)

# Fields the cache stores per review. Keep in sync with db.save_features_batch.
CACHED_FIELDS = (
    "polarity", "subjectivity", "aspects", "entities",
    "emotion", "urgency", "embedding",
)

# Noise words to filter out of aspect extraction
STOPWORDS = {
    "app", "it", "this", "i", "we", "they", "you", "he", "she",
    "thing", "everything", "nothing", "anything", "them", "time",
    "what", "things", "something", "someone", "way", "lot", "bit",
    "kind", "sort", "stuff", "issue", "problem", "one", "two", "that"
}

# Entity labels kept for competitor/product detection
ENTITY_LABELS = {"ORG", "PRODUCT"}

# Common English words spaCy sometimes mis-tags as ORG/PRODUCT, especially
# when they appear in ALL-CAPS for emphasis (e.g. "I NEVER got my order").
NER_STOPWORDS = {
    "app", "apps", "the app", "this app",
    "never", "good", "great", "bad", "help", "please",
    "thanks", "thank you", "ok", "okay", "hi", "hello", "yes", "no",
    "wow", "omg", "wtf", "lol", "lmao", "yeah", "nope", "nah",
    "fine", "sure", "well", "ugh", "eh", "huh", "nothing", "something",
    # UI button / CTA labels that surface in ALL-CAPS and get mistagged as ORG
    "cancel", "buy", "pay", "submit", "click", "tap",
    # Acronyms / single-word product mentions spaCy mis-tags as ORG
    # (observed on the Amazon corpus: OTP / Newest / Tablet appearing as
    # entities even though they're domain words, not company names).
    "otp", "newest", "tablet", "tablets",
}

# Keywords that signal a concrete technical/bug issue worth escalating
BUG_KEYWORDS = {
    "crash", "bug", "broken", "freeze", "frozen", "error", "errors",
    "glitch", "stuck", "fail", "fails", "failed", "loading", "loads",
    "not working", "doesn't work", "won't", "wont", "can't", "cant",
    "cannot", "unable", "black screen", "blank", "lag", "laggy",
}


def aspect_names(review):
    """Return aspect name strings from a review dict.

    Handles both the Phase 3+ format list[{aspect, polarity, confidence}]
    and the legacy list[str] format so code that calls this works across
    cached and freshly-computed reviews.
    """
    return [
        a["aspect"] if isinstance(a, dict) else a
        for a in (review.get("aspects") or [])
    ]


def sentiment_features(text):
    """Extract polarity and subjectivity from review text using TextBlob."""
    if not text:
        return {"polarity": None, "subjectivity": None}

    blob = TextBlob(text)
    return {
        "polarity": round(blob.sentiment.polarity, 4),
        "subjectivity": round(blob.sentiment.subjectivity, 4),
    }


def spacy_features(text, brand_stopwords=frozenset()):
    """
    Single spaCy pass that extracts both aspects (noun phrases) and
    named entities (organizations, products). Combining avoids running
    nlp() twice per review.
    """
    if not text:
        return {"aspects": [], "entities": []}

    doc = nlp(text)

    # Aspects: lemma of each noun-chunk root
    aspects = []
    seen = set()
    for chunk in doc.noun_chunks:
        aspect = chunk.root.lemma_.lower()
        if aspect in STOPWORDS or len(aspect) <= 2 or not aspect.isalpha():
            continue
        if aspect in brand_stopwords:
            continue
        if aspect in seen:
            continue
        seen.add(aspect)
        aspects.append(aspect)

    # Entities: ORG / PRODUCT mentions, with brand + NER noise filtered out
    entities = []
    seen_ents = set()
    for ent in doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue
        name = ent.text.strip()
        key = name.lower()
        if not name or len(key) < 3 or key in seen_ents:
            continue
        if key in NER_STOPWORDS:
            continue
        # Substring match so "Amazon Shopping" is filtered when brand is "amazon"
        if any(b and b in key for b in brand_stopwords):
            continue
        seen_ents.add(key)
        entities.append({"text": name, "label": ent.label_})

    return {"aspects": aspects, "entities": entities}


def embedding_features(texts):
    """Batch-encode texts into 768-dim semantic vectors."""
    if not texts:
        return np.array([])
    return embedding_model.encode(texts, show_progress_bar=True, batch_size=INFERENCE_BATCH)


def emotion_features(texts):
    """
    Classify each text into a single dominant emotion label.
    Uses a pretrained DistilRoBERTa model (anger, disgust, fear,
    joy, neutral, sadness, surprise).
    """
    if not texts:
        return []

    safe = [(t or ".")[:1000] for t in texts]
    results = emotion_classifier(safe, batch_size=INFERENCE_BATCH, truncation=True, max_length=128)

    labels = []
    for original, r in zip(texts, results):
        if not original:
            labels.append(None)
            continue
        # With top_k=1 the pipeline returns [{"label":..., "score":...}]
        item = r[0] if isinstance(r, list) and r else r
        labels.append(item["label"] if item else None)
    return labels


def absa_features(texts, aspects_lists):
    """Score each (text, aspect) pair with the DeBERTa ABSA model.

    Returns list[list[{aspect, polarity, confidence}]] — one inner list per
    review, preserving the order of aspects_lists. Reviews with no aspects
    get an empty list.

    Input format: the model is queried with `text_pair` so the tokenizer
    encodes the pair as [CLS] review [SEP] aspect [SEP], matching its
    fine-tuning setup. We cap at ABSA_MAX_ASPECTS per review so a review
    with an unusually large noun-chunk list doesn't dominate inference time.

    Polarity mapping: Positive label → +confidence, Negative → -confidence,
    Neutral → 0.0.
    """
    flat_inputs = []
    flat_index = []  # (review_idx, aspect_str) for result reconstruction

    for rev_idx, (text, aspects) in enumerate(zip(texts, aspects_lists)):
        safe_text = (text or "")[:512]
        for aspect in (aspects or [])[:ABSA_MAX_ASPECTS]:
            if aspect and safe_text:
                flat_inputs.append({"text": safe_text, "text_pair": str(aspect)})
                flat_index.append((rev_idx, aspect))

    # Manual batching with tqdm so CPU users (where ABSA dominates total time)
    # can see progress instead of staring at a blank screen for 25 minutes.
    # The progress bar is suppressed on GPU since the whole step finishes
    # in seconds and a flickering bar there is just noise.
    raw_results = []
    if flat_inputs:
        for i in tqdm(
            range(0, len(flat_inputs), INFERENCE_BATCH),
            desc="    ABSA",
            disable=USE_GPU,
            unit="batch",
        ):
            batch = flat_inputs[i:i + INFERENCE_BATCH]
            batch_results = absa_classifier(
                batch, batch_size=INFERENCE_BATCH, truncation=True, max_length=256
            )
            raw_results.extend(batch_results)

    output = [[] for _ in range(len(texts))]
    for (rev_idx, aspect), pred in zip(flat_index, raw_results):
        item = pred[0] if isinstance(pred, list) and pred else pred
        label = (item.get("label") or "neutral").lower() if item else "neutral"
        score = item.get("score", 0.0) if item else 0.0

        if "positive" in label:
            polarity = score
        elif "negative" in label:
            polarity = -score
        else:
            polarity = 0.0

        output[rev_idx].append({
            "aspect": aspect,
            "polarity": round(polarity, 3),
            "confidence": round(score, 3),
        })

    return output


def urgency_score(review):
    """
    Heuristic 0-1 score for how actionable a review is. Combines concrete
    bug signals, low rating, low subjectivity (fact-based), specific aspects,
    and non-trivial length. Not ML — intentionally interpretable.
    """
    body = (review.get("body") or "").lower()
    if not body:
        return 0.0

    score = 0.0
    if any(kw in body for kw in BUG_KEYWORDS):
        score += 0.4

    rating = review.get("rating")
    if rating is not None and rating <= 2:
        score += 0.3

    subj = review.get("subjectivity")
    if subj is not None and subj < 0.5:
        score += 0.15

    if len(review.get("aspects") or []) >= 2:
        score += 0.1

    if len(body) >= 50:
        score += 0.05

    return round(min(score, 1.0), 3)


def cluster_themes(embeddings, n_clusters=10):
    """Cluster embeddings into theme labels with KMeans."""
    if len(embeddings) == 0:
        return []

    n_clusters = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings).tolist()


def select_k(embeddings, k_range=DEFAULT_K_RANGE, sample_size=SILHOUETTE_SAMPLE_SIZE,
             random_state=42, verbose=True):
    """
    Sweep K and return the value with the highest silhouette score.

    For tiny corpora (< 100 reviews) the sweep is skipped and a small fixed K
    is returned — silhouette is unreliable when each cluster has only a
    handful of points.

    Silhouette is O(n²). On corpora larger than `sample_size`, scoring is
    computed on a random subsample for speed; the clustering itself still
    runs over all points.
    """
    n = len(embeddings)
    if n < 100:
        return min(8, max(2, n // 5))

    k_min, k_max_exclusive = k_range
    # never produce clusters smaller than ~5 reviews on average
    k_max = min(k_max_exclusive - 1, n // 5)
    if k_max <= k_min:
        return k_min

    scores = []
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(embeddings)
        score = silhouette_score(
            embeddings, labels,
            sample_size=min(sample_size, n),
            random_state=random_state,
        )
        scores.append((k, score))
        if verbose:
            print(f"    K={k}: silhouette={score:.3f}")

    best_k, best_score = max(scores, key=lambda x: x[1])
    if verbose:
        print(f"  Selected K={best_k} (silhouette={best_score:.3f})")
    return best_k


def _overlap_coefficient(a, b):
    """|A ∩ B| / min(|A|, |B|). Tolerant of differing set sizes."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _aspect_doc_freq(reviews):
    """How many reviews mention each aspect — corpus document frequency for TF-IDF."""
    df = Counter()
    for r in reviews:
        for a in set(aspect_names(r)):
            df[a] += 1
    return df


def _distinctive_aspects(cluster_reviews, corpus_df, total_reviews, k=8, min_count=3):
    """
    Top-k aspects ranked by TF-IDF distinctiveness within this cluster:
    `cluster_count * log(total_reviews / corpus_df)`. This pushes generic
    high-frequency words ('item', 'service', 'order') down — they appear in
    most clusters so their IDF is low — and surfaces cluster-specific cores
    ('tablet', 'delivery', 'mode') which are exactly what should drive merges.

    Falls back to raw frequency on small clusters (< 20 reviews) where IDF
    is statistically unstable.
    """
    cluster_count = Counter()
    for r in cluster_reviews:
        for a in set(aspect_names(r)):
            cluster_count[a] += 1

    if not cluster_count:
        return []

    if len(cluster_reviews) < 20:
        return [a for a, _ in cluster_count.most_common(k)]

    scored = []
    for aspect, count in cluster_count.items():
        if count < min_count:
            continue
        df = corpus_df.get(aspect, 1)
        idf = math.log(total_reviews / df) if df else 0.0
        # 0 IDF means this aspect appears in every review → completely
        # uninformative for distinguishing clusters. Drop it.
        if idf <= 0:
            continue
        scored.append((aspect, count * idf))
    scored.sort(key=lambda x: -x[1])
    distinctive = [a for a, _ in scored[:k]]
    return distinctive or [a for a, _ in cluster_count.most_common(k)]


def merge_similar_clusters(reviews, threshold=MERGE_THRESHOLD, top_k=MERGE_TOP_K, verbose=True):
    """
    Collapse clusters whose top-K aspect sets overlap above `threshold` (overlap
    coefficient). Reassigns `theme_cluster` on each review and renumbers the
    surviving clusters to a contiguous 0..N-1 range.

    Single-pass union-find: pairs are evaluated once on the original cluster
    aspects. Iterative-to-convergence would catch transitive cases (A~B, B~C
    where A and C aren't directly similar) but adds little in practice and
    risks over-merging.
    """
    groups = defaultdict(list)
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is not None:
            groups[cid].append(r)

    if len(groups) < 2:
        return reviews

    # Use TF-IDF distinctiveness, not raw frequency. Otherwise the top-K
    # aspects of every cluster in an e-commerce corpus get dominated by
    # filler words ('item', 'service', 'order') that appear everywhere,
    # which drives false merges between clusters whose actual cores
    # (tablet vs delivery vs payment) are completely different.
    corpus_df = _aspect_doc_freq(reviews)
    total_reviews = len(reviews)
    aspect_sets = {
        cid: set(_distinctive_aspects(g, corpus_df, total_reviews, k=top_k))
        for cid, g in groups.items()
    }

    parent = {cid: cid for cid in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Use the smaller cid as the canonical root for stable output
            root = min(ra, rb)
            other = max(ra, rb)
            parent[other] = root

    cids = sorted(groups.keys())
    merges = []
    for i, c1 in enumerate(cids):
        for c2 in cids[i + 1:]:
            score = _overlap_coefficient(aspect_sets[c1], aspect_sets[c2])
            if score >= threshold:
                merges.append((c1, c2, score))
                union(c1, c2)

    if not merges:
        if verbose:
            print("  No clusters merged.")
        return reviews

    # Reassign every review to its component root
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is not None:
            r["theme_cluster"] = find(cid)

    # Renumber surviving clusters to a contiguous 0..N-1 range
    surviving = sorted({r["theme_cluster"] for r in reviews if r.get("theme_cluster") is not None})
    remap = {cid: i for i, cid in enumerate(surviving)}
    for r in reviews:
        cid = r.get("theme_cluster")
        if cid is not None:
            r["theme_cluster"] = remap[cid]

    if verbose:
        print(f"  Merged {len(merges)} cluster pair(s):")
        for c1, c2, score in merges:
            shared = aspect_sets[c1] & aspect_sets[c2]
            print(f"    {c1} ↔ {c2} (overlap={score:.2f}, shared: {', '.join(sorted(shared))})")
        print(f"  Final cluster count: {len(surviving)}")
    return reviews


def run_pipeline(reviews, n_clusters=None, k_range=DEFAULT_K_RANGE,
                 merge_threshold=MERGE_THRESHOLD, merge_top_k=MERGE_TOP_K,
                 brand_stopwords=None, use_cache=True):
    """
    Run the full feature engineering pipeline on a list of review dicts.
    Features attached to each review: polarity, subjectivity, aspects,
    entities, emotion, urgency, embedding, theme_cluster.

    Clustering:
        - n_clusters=None (default): K is chosen by silhouette score over k_range.
        - n_clusters=int: forces a fixed K (useful for reproducibility).
        After KMeans, near-duplicate clusters whose top-aspect sets overlap
        above merge_threshold are collapsed in a single union-find pass.

    With use_cache=True (default), per-review features are loaded from the
    SQLite feature cache when available and only the missing ones are
    recomputed. Clustering always re-runs on the full embedding set since
    cluster IDs are not stable across runs.
    """
    brand_stopwords = frozenset(s.lower() for s in (brand_stopwords or []))
    print(f"Running feature engineering on {len(reviews)} reviews...")

    cache_keys = [compute_cache_key(r) for r in reviews]
    cached = {}
    if use_cache:
        cached = load_features_batch(
            cache_keys, FEATURE_SCHEMA_VERSION, EMBEDDER_MODEL_NAME
        )
        for r, key in zip(reviews, cache_keys):
            entry = cached.get(key)
            if entry is None:
                continue
            for field in CACHED_FIELDS:
                r[field] = entry.get(field)
        print(f"  Cache: {len(cached)} hits / {len(reviews) - len(cached)} misses")
    else:
        print("  Cache: disabled (--no-cache)")

    to_compute_idx = [i for i, key in enumerate(cache_keys) if key not in cached]
    to_compute = [reviews[i] for i in to_compute_idx]

    if to_compute:
        print(f"  Computing features for {len(to_compute)} reviews...")

        print("    polarity / subjectivity")
        for r in to_compute:
            r.update(sentiment_features(r.get("body", "")))

        print("    aspects / entities (spaCy)")
        for r in to_compute:
            r.update(spacy_features(r.get("body", ""), brand_stopwords))

        print("    urgency")
        for r in to_compute:
            r["urgency"] = urgency_score(r)

        texts = [r.get("body", "") or "" for r in to_compute]

        # ABSA: score each (text, aspect) pair. At this point aspects is still
        # list[str] from spaCy — urgency already consumed it, so we can safely
        # upgrade to list[dict] here before saving to cache.
        print("    aspect sentiment (ABSA)")
        raw_aspect_lists = [list(r.get("aspects") or []) for r in to_compute]
        absa_results = absa_features(texts, raw_aspect_lists)
        for r, scored_aspects in zip(to_compute, absa_results):
            r["aspects"] = scored_aspects

        print("    embeddings")
        embeddings = embedding_features(texts)
        for i, r in enumerate(to_compute):
            r["embedding"] = (
                embeddings[i].tolist() if len(embeddings) > 0 else None
            )

        print("    emotions")
        emotions = emotion_features(texts)
        for r, e in zip(to_compute, emotions):
            r["emotion"] = e

        if use_cache:
            items = [(cache_keys[i], reviews[i]) for i in to_compute_idx]
            save_features_batch(
                items, FEATURE_SCHEMA_VERSION, EMBEDDER_MODEL_NAME
            )
            print(f"  Saved {len(items)} new entries to cache")
    else:
        print("  All features served from cache; skipping model inference.")

    # Clustering always runs on the full corpus. None embeddings (very rare;
    # would only happen if a cached row was corrupted) are excluded gracefully.
    valid = [(i, r) for i, r in enumerate(reviews) if r.get("embedding") is not None]
    for r in reviews:
        r["theme_cluster"] = None

    if valid:
        embedding_matrix = np.array([r["embedding"] for _, r in valid])

        if n_clusters is None:
            print("  Selecting K via silhouette score...")
            k = select_k(embedding_matrix, k_range=k_range)
        else:
            k = n_clusters
            print(f"  Using fixed K={k}")

        print(f"  Clustering into {k} themes...")
        labels = cluster_themes(embedding_matrix, n_clusters=k)
        if labels:
            for (idx, _), lab in zip(valid, labels):
                reviews[idx]["theme_cluster"] = lab

        print("  Merging near-duplicate clusters...")
        merge_similar_clusters(
            reviews, threshold=merge_threshold, top_k=merge_top_k
        )

    print(f"Done. Features ready for {len(reviews)} reviews.")
    return reviews


def validate_features(reviews):
    """Print basic validation stats to confirm features are meaningful."""
    from collections import Counter, defaultdict

    print("\n--- Feature Validation ---")

    print("\nAverage polarity by star rating:")
    polarity_by_rating = defaultdict(list)
    for r in reviews:
        if r.get("rating") and r.get("polarity") is not None:
            polarity_by_rating[r["rating"]].append(r["polarity"])

    for rating in sorted(polarity_by_rating.keys()):
        values = polarity_by_rating[rating]
        avg = round(sum(values) / len(values), 4)
        print(f"  {rating} stars: avg polarity = {avg} ({len(values)} reviews)")

    print("\nTop aspects in 1-star reviews:")
    one_star_aspects = []
    for r in reviews:
        if r.get("rating") == 1:
            one_star_aspects.extend(aspect_names(r))
    for aspect, count in Counter(one_star_aspects).most_common(10):
        print(f"  '{aspect}': {count} mentions")

    print("\nEmotion distribution:")
    emotions = [r.get("emotion") for r in reviews if r.get("emotion")]
    for emotion, count in Counter(emotions).most_common():
        print(f"  {emotion}: {count}")

    print("\nSample reviews per cluster (first 3 clusters):")
    clusters = defaultdict(list)
    for r in reviews:
        if r.get("theme_cluster") is not None:
            clusters[r["theme_cluster"]].append(r.get("body", "")[:100])

    for cluster_id in sorted(clusters.keys())[:3]:
        print(f"\n  Cluster {cluster_id}:")
        for sample in clusters[cluster_id][:3]:
            print(f"    - {sample}")
