from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline as hf_pipeline
import numpy as np

# Load models once at module level so they are not reloaded on every call
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
emotion_classifier = hf_pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
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
}

# Keywords that signal a concrete technical/bug issue worth escalating
BUG_KEYWORDS = {
    "crash", "bug", "broken", "freeze", "frozen", "error", "errors",
    "glitch", "stuck", "fail", "fails", "failed", "loading", "loads",
    "not working", "doesn't work", "won't", "wont", "can't", "cant",
    "cannot", "unable", "black screen", "blank", "lag", "laggy",
}


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
    """Batch-encode texts into 384-dim semantic vectors."""
    if not texts:
        return np.array([])
    return embedding_model.encode(texts, show_progress_bar=True, batch_size=64)


def emotion_features(texts):
    """
    Classify each text into a single dominant emotion label.
    Uses a pretrained DistilRoBERTa model (anger, disgust, fear,
    joy, neutral, sadness, surprise).
    """
    if not texts:
        return []

    safe = [(t or ".")[:1000] for t in texts]
    results = emotion_classifier(safe, batch_size=32, truncation=True, max_length=128)

    labels = []
    for original, r in zip(texts, results):
        if not original:
            labels.append(None)
            continue
        # With top_k=1 the pipeline returns [{"label":..., "score":...}]
        item = r[0] if isinstance(r, list) and r else r
        labels.append(item["label"] if item else None)
    return labels


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


def run_pipeline(reviews, n_clusters=10, brand_stopwords=None):
    """
    Run the full feature engineering pipeline on a list of review dicts.
    Features attached to each review: polarity, subjectivity, aspects,
    entities, emotion, urgency, embedding, theme_cluster.
    """
    brand_stopwords = frozenset(s.lower() for s in (brand_stopwords or []))
    print(f"Running feature engineering on {len(reviews)} reviews...")

    print("  Extracting polarity and subjectivity...")
    for review in reviews:
        review.update(sentiment_features(review.get("body", "")))

    print("  Extracting aspects and entities (spaCy)...")
    for review in reviews:
        review.update(spacy_features(review.get("body", ""), brand_stopwords))

    print("  Scoring urgency...")
    for review in reviews:
        review["urgency"] = urgency_score(review)

    texts = [review.get("body", "") or "" for review in reviews]

    print("  Generating embeddings...")
    embeddings = embedding_features(texts)

    print("  Classifying emotions...")
    emotions = emotion_features(texts)
    for r, e in zip(reviews, emotions):
        r["emotion"] = e

    print(f"  Clustering into {n_clusters} themes...")
    labels = cluster_themes(embeddings, n_clusters=n_clusters)

    for i, review in enumerate(reviews):
        review["theme_cluster"] = labels[i] if labels else None
        review["embedding"] = embeddings[i].tolist() if len(embeddings) > 0 else None

    print(f"Done. Features extracted for {len(reviews)} reviews.")
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
            one_star_aspects.extend(r.get("aspects", []))
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
