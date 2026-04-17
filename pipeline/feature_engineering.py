from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Load models once at module level so they are not reloaded on every call
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Noise words to filter out of aspect extraction
STOPWORDS = {
    "app", "it", "this", "i", "we", "they", "you", "he", "she",
    "thing", "everything", "nothing", "anything", "them", "time",
    "what", "things", "something", "someone", "way", "lot", "bit",
    "kind", "sort", "stuff", "issue", "problem", "one", "two", "that"
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


def aspect_features(text):
    """Extract noun phrases from review text using spaCy."""
    if not text:
        return {"aspects": []}

    doc = nlp(text)
    aspects = []

    for chunk in doc.noun_chunks:
        # Take the root noun of the phrase, lowercased
        aspect = chunk.root.text.lower()
        if aspect not in STOPWORDS and len(aspect) > 2:
            aspects.append(aspect)

    # Deduplicate while preserving order
    seen = set()
    unique_aspects = []
    for a in aspects:
        if a not in seen:
            seen.add(a)
            unique_aspects.append(a)

    return {"aspects": unique_aspects}


def embedding_features(texts):
    """
    Generate semantic embeddings for a list of texts.
    Returns a numpy array of shape (n_reviews, 384).
    Processes in batch for efficiency.
    """
    if not texts:
        return np.array([])

    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


def cluster_themes(embeddings, n_clusters=10):
    """
    Cluster embeddings into themes using KMeans.
    Returns a list of cluster labels (integers).
    """
    if len(embeddings) == 0:
        return []

    # Adjust clusters if we have fewer reviews than requested clusters
    n_clusters = min(n_clusters, len(embeddings))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels.tolist()


def run_pipeline(reviews, n_clusters=10):
    """
    Run the full feature engineering pipeline on a list of review dicts.
    Each review dict must have at least a 'body' and 'review_id' field.
    Returns a list of feature dicts.
    """
    print(f"Running feature engineering on {len(reviews)} reviews...")

    # Step 1: Sentiment features
    print("  Extracting polarity and subjectivity...")
    for review in reviews:
        sentiment = sentiment_features(review.get("body", ""))
        review.update(sentiment)

    # Step 2: Aspect features
    print("  Extracting aspects...")
    for review in reviews:
        aspects = aspect_features(review.get("body", ""))
        review.update(aspects)

    # Step 3: Embeddings (batch)
    print("  Generating embeddings...")
    texts = [review.get("body", "") or "" for review in reviews]
    embeddings = embedding_features(texts)

    # Step 4: Clustering
    print(f"  Clustering into {n_clusters} themes...")
    labels = cluster_themes(embeddings, n_clusters=n_clusters)

    # Step 5: Attach embedding and cluster label to each review
    for i, review in enumerate(reviews):
        review["theme_cluster"] = labels[i] if labels else None
        review["embedding"] = embeddings[i].tolist() if len(embeddings) > 0 else None

    print(f"Done. Features extracted for {len(reviews)} reviews.")
    return reviews


def validate_features(reviews):
    """
    Print basic validation stats to confirm features are meaningful.
    """
    print("\n--- Feature Validation ---")

    # Polarity by star rating
    print("\nAverage polarity by star rating:")
    from collections import defaultdict
    polarity_by_rating = defaultdict(list)
    for r in reviews:
        if r.get("rating") and r.get("polarity") is not None:
            polarity_by_rating[r["rating"]].append(r["polarity"])

    for rating in sorted(polarity_by_rating.keys()):
        values = polarity_by_rating[rating]
        avg = round(sum(values) / len(values), 4)
        print(f"  {rating} stars: avg polarity = {avg} ({len(values)} reviews)")

    # Most common aspects in 1-star reviews
    print("\nTop aspects in 1-star reviews:")
    from collections import Counter
    one_star_aspects = []
    for r in reviews:
        if r.get("rating") == 1:
            one_star_aspects.extend(r.get("aspects", []))

    top_aspects = Counter(one_star_aspects).most_common(10)
    for aspect, count in top_aspects:
        print(f"  '{aspect}': {count} mentions")

    # Sample reviews per cluster
    print("\nSample reviews per cluster (first 3 clusters):")
    from collections import defaultdict
    clusters = defaultdict(list)
    for r in reviews:
        if r.get("theme_cluster") is not None:
            clusters[r["theme_cluster"]].append(r.get("body", "")[:100])

    for cluster_id in sorted(clusters.keys())[:3]:
        print(f"\n  Cluster {cluster_id}:")
        for sample in clusters[cluster_id][:3]:
            print(f"    - {sample}")