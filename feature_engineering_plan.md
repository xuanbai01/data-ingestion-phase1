# Feature Engineering Pipeline: Research and Design Plan

**Author:** Xuan Bai 
**Date:** April 2026  
**Project:** Data Ingestion System, Phase II

---

## 1. Overview

Phase I of this project built a pipeline that collects, cleans, and stores raw app reviews from Google Play Store. The next step is to transform that raw text into structured, model-ready features that downstream ML systems can actually learn from.

Raw review text on its own is not directly useful for modeling. A review like *"the app keeps crashing on login and it's so frustrating"* is just a string. This pipeline converts it into structured signals: what sentiment it carries, what part of the product it references, and how it relates semantically to other reviews. The result is something a model can actually work with.

---

## 2. Feature Selection and Rationale

Three features were selected for this prototype. Each was chosen for a specific reason tied to downstream ML value.

### 2.1 Polarity and Subjectivity

**What it is:**  
Polarity measures how positive or negative a review is, on a scale from -1.0 (very negative) to 1.0 (very positive). Subjectivity measures whether the review is fact-based or opinion-based, on a scale from 0.0 (objective) to 1.0 (subjective).

**Why it's useful:**  
Polarity alone does not distinguish between a bug report and an emotional complaint. Both can score equally negative. Subjectivity separates these two cases. A low-subjectivity negative review (*"the app crashes on startup"*) signals a real technical issue that engineering should act on. A high-subjectivity negative review (*"this app is absolutely terrible"*) signals a user experience problem that customer success should handle.

For downstream ML, this distinction enables more targeted modeling, for example training separate classifiers for technical issues vs. experience complaints.

**Tool:** TextBlob

---

### 2.2 Aspect Extraction

**What it is:**  
Aspect extraction identifies which part of the product a review is discussing. Using part-of-speech parsing, we extract noun phrases from review text. These nouns typically correspond to product features being discussed (e.g., "login", "UI", "notifications", "battery", "checkout").

**Why it's useful:**  
Without aspect extraction, you know a review is negative but not what it is negative about. With it, you can aggregate across reviews to identify which product areas have the most complaints. For example, if 800 out of 10,000 reviews mention "login" negatively, that is an actionable signal that a specific feature needs attention.

For ML, extracted aspects become categorical features that let models learn patterns like "reviews mentioning UI tend to be 1-2 stars", which is much richer signal than raw text.

**Tool:** spaCy (en_core_web_sm)

---

### 2.3 Semantic Embeddings

**What it is:**  
An embedding is a numeric vector that represents the meaning of a piece of text. The key property is that semantically similar texts produce numerically similar vectors. For example, "laggy", "slow", and "freezing" would all produce vectors close to each other, even though they share no words.

**Why it's useful:**  
Raw text cannot be fed directly into most ML models since they require numeric input. Embeddings convert text into numbers while preserving semantic meaning. By clustering similar embeddings together, we can automatically discover themes in review data (e.g., "performance issues", "payment problems", "login failures") without manual labeling.

This is especially valuable at scale. With 10k+ reviews, manual theme identification is not feasible, but clustering on embeddings makes it automatic.

**Tool:** sentence-transformers (all-MiniLM-L6-v2)

---

## 3. Pipeline Design

```
Raw Review Text
      |
      |---> TextBlob ------------> polarity, subjectivity
      |
      |---> spaCy --------------> extracted aspects (noun phrases)
      |
      └---> sentence-transformers -> embedding vector (384 dimensions)
                |
                └---> KMeans clustering ---> theme label (0-N)
                      |
                      v
          Structured Feature Row:
          {
            review_id, body, polarity, subjectivity,
            aspects, embedding, theme_cluster
          }
```

Each review passes through all three modules independently. The outputs are combined into a single structured row per review, which is then stored and exported as a model-ready dataset.

### Module Breakdown

| Module | Input | Output |
|---|---|---|
| `sentiment_features()` | review text | polarity (float), subjectivity (float) |
| `aspect_features()` | review text | list of noun phrases |
| `embedding_features()` | review text | 384-dim vector |
| `cluster_themes()` | all embeddings | theme label per review |
| `run_pipeline()` | list of reviews | structured feature dataset |

---

## 4. Validation Plan

To confirm that the extracted features are actually meaningful, the following checks will be applied:

**Polarity validation:**  
Compare average polarity across star rating groups. 1-star reviews should have significantly lower polarity than 5-star reviews. If they do not, the polarity signal is not useful.

**Aspect validation:**  
Inspect the most frequently extracted aspects across 1-star reviews. Check whether they correspond to known product pain points (e.g., "login", "crash", "loading"). If aspects are mostly generic words, the extraction needs tuning.

**Embedding and clustering validation:**  
Manually inspect 5-10 reviews from each cluster and verify they share a coherent theme. A good cluster should contain reviews about the same topic. A bad cluster will have unrelated reviews grouped together.

---

## 5. Tools and Libraries

| Library | Purpose | Why chosen |
|---|---|---|
| TextBlob | Polarity and subjectivity | Lightweight, interpretable, no training required |
| spaCy | Aspect extraction via POS tagging | Fast, production-grade NLP, good noun phrase detection |
| sentence-transformers | Semantic embeddings | State-of-the-art sentence embeddings, easy to use |
| scikit-learn | KMeans clustering | Standard, well-documented, integrates cleanly |

---

## 6. Scope and Limitations

This is a first prototype, not a production system. Known limitations:

- Aspect extraction relies on noun phrases which can include noise (e.g., "I", "it"). Filtering will be applied but not exhaustively tuned.
- Brand names (e.g., "amazon") surface as top aspects in both positives and negatives because users mention the brand in almost any review. This is noise, not signal — a future iteration should add a per-app brand stopword list.
- Clustering requires a fixed number of clusters (K) chosen manually. Optimal K will be estimated but not rigorously validated.
- Embeddings are computed per review independently with no cross-review context.
- Theme clusters dominated by very short reviews (e.g., "Good", "great") produce weak representative examples even after dedupe. The cluster label (top aspects) remains informative.

These are acceptable tradeoffs for a prototype. A future iteration could address them with more sophisticated NLP models (e.g., fine-tuned BERT for aspect extraction) and automated cluster selection (e.g., elbow method).

---

## 7. Extensions (v2)

After the v1 prototype validated polarity, aspects, and clustering as meaningful signals, three additional features were layered on to make the pipeline output more actionable:

### 7.1 Emotion Classification
**What:** Each review is classified into one dominant emotion (anger, disgust, fear, joy, neutral, sadness, surprise) using `j-hartmann/emotion-english-distilroberta-base`.
**Why:** Polarity tells us *how* negative a review is; emotion tells us *what kind* of negative it is. Anger-heavy feedback signals churn risk; sadness-heavy feedback signals disappointment with a specific feature; fear-heavy feedback often indicates trust/security concerns. These are different operational responses.

### 7.2 Urgency / Actionability Score
**What:** A heuristic 0-1 score combining bug-related keywords, low rating, low subjectivity, aspect specificity, and review length. Not ML — intentionally interpretable.
**Why:** Sentiment alone does not tell a product team what to do first. A 5-item score lets us surface the most actionable bug reports above generic emotional complaints, directly answering "what should we look at this sprint?"

### 7.3 Named Entity Recognition
**What:** spaCy extracts `ORG` and `PRODUCT` entities from each review, with the host brand filtered out.
**Why:** Surfaces competitor mentions ("switched to Target", "ebay is better") and specific product references that aspect extraction alone misses. Useful for competitive-intelligence and product-manager signal.

All three features plug into the existing `run_pipeline` and the summarizer generates a dedicated section for each.

---

## 8. Next Steps

1. Implement `pipeline/feature_engineering.py` with the four modules above
2. Run on the existing 10k Amazon reviews
3. Validate outputs against the criteria in Section 4
4. Export feature dataset as CSV for downstream use
