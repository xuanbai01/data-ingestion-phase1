# App Review Ingestion & Feature Engineering Pipeline

An end-to-end pipeline that scrapes app store reviews, cleans and stores them in a relational database, extracts model-ready NLP features, and generates human-readable analytical reports.

Built in two phases:

- **Phase I** — data ingestion: scraping → cleaning → storage → CSV export
- **Phase II** — feature engineering: raw review text → structured signals (sentiment, aspects, emotion, urgency, entities, embeddings, themes) → summary report

---

## Project Structure

```
data_ingestion/
├── scraper/
│   └── play_scraper.py        # Google Play scraper with pagination
├── database/
│   └── db.py                  # SQLite schema + dedup logic
├── pipeline/
│   ├── cleaner.py             # Normalize whitespace, dates, ratings
│   ├── feature_engineering.py # Sentiment, aspects, emotion, urgency, NER, embeddings, themes
│   ├── summarizer.py          # Markdown report generator
│   ├── exporter.py            # CSV read/write
│   └── logger.py
├── logs/                      # Per-run execution logs (auto-generated)
├── exports/                   # Per-run CSV backups (auto-generated)
├── reports/                   # Per-run markdown summaries (auto-generated)
├── reviews.db                 # SQLite database (auto-generated)
├── feature_engineering_plan.md  # Design rationale for Phase II
├── main.py
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On first run, HuggingFace will download the emotion-classification model (~330 MB).

---

## Usage

### Full pipeline (scrape → features → report)
```bash
python main.py --apps amazon --count 10000 --export
```

### Multi-app
```bash
python main.py --apps amazon ebay walmart --count 10000 --export
```

### Iterate on feature engineering without re-scraping
Re-use a previously exported cleaned CSV to avoid hitting Google Play again:
```bash
python main.py --apps amazon --from-csv exports/amazon_20260422_023736.csv --export
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--apps` | `amazon` | One or more app keys to process |
| `--count` | `10000` | Reviews to fetch per app (ignored with `--from-csv`) |
| `--export` | `False` | Write results to `exports/` |
| `--from-csv` | `None` | Load reviews from an existing cleaned CSV; skip scrape + DB |

### Supported Apps

| Key | App | Package ID |
|---|---|---|
| `amazon` | Amazon Shopping | `com.amazon.mShop.android.shopping` |
| `ebay` | eBay | `com.ebay.mobile` |
| `walmart` | Walmart | `com.walmart.android` |

Add a new app by editing the `APPS` dict in [main.py](main.py).

---

## Pipeline Overview

```
Scrape ─► Clean ─► Store ─► Feature Engineering ─► Summary Report
                     │                                   │
                     ▼                                   ▼
                  SQLite                         reports/{app}_{ts}.md
                                                 exports/{app}_features_{ts}.csv
```

Each app runs independently. Reviews are deduped by Google Play `review_id` with an MD5 hash fallback, so re-running the pipeline adds only new reviews to the database.

---

## Phase II — Feature Engineering

For every review the pipeline produces:

| Feature | Type | Tool | Purpose |
|---|---|---|---|
| `polarity` | float [-1, 1] | TextBlob | How positive/negative |
| `subjectivity` | float [0, 1] | TextBlob | Fact-based vs opinion-based |
| `aspects` | list[str] | spaCy | Product parts being discussed (lemmatized noun roots) |
| `entities` | list[{text, label}] | spaCy NER | ORG / PRODUCT mentions (competitors, platforms) |
| `emotion` | str | DistilRoBERTa | anger / disgust / fear / joy / neutral / sadness / surprise |
| `urgency` | float [0, 1] | heuristic | Actionability score (bug keywords + low rating + low subjectivity + aspect specificity) |
| `embedding` | list[float, 384] | sentence-transformers (all-MiniLM-L6-v2) | Semantic vector |
| `theme_cluster` | int | KMeans over embeddings | Auto-discovered theme label |

Design rationale and validation criteria are documented in [feature_engineering_plan.md](feature_engineering_plan.md).

### Summary Report

Every run writes a markdown report to `reports/{app}_{timestamp}.md` with:

- **Overall Sentiment** — avg rating, avg polarity, % positive/negative, objective-vs-subjective negative split
- **Emotion Distribution** — counts per dominant emotion
- **Top Issues** — most mentioned aspects in 1–2 star reviews (with avg polarity & rating)
- **Top Positives** — most mentioned aspects in 4–5 star reviews
- **Most Urgent Reviews** — top 10 reviews ranked by urgency score
- **Mentioned Entities** — ORG / PRODUCT entities, brand-filtered
- **Themes** — each auto-discovered cluster with top aspects, stats, and representative reviews

### Extending the Feature Set

The pipeline is organized so new features plug in without touching existing ones:

1. Add a new feature function in [pipeline/feature_engineering.py](pipeline/feature_engineering.py) that reads `review["body"]` (or other existing fields) and returns a dict of new keys.
2. Call it in `run_pipeline()` at the appropriate step.
3. Optionally add a new section to [pipeline/summarizer.py](pipeline/summarizer.py) that reads those new keys and register it in `generate_report()`.

---

## Database Schema

### `companies`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| name | TEXT | App display name |
| slug | TEXT UNIQUE | Google Play package ID |
| created_at | TEXT | Timestamp |

### `reviews`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| company_id | INTEGER | FK → companies.id |
| review_id | TEXT UNIQUE | Google Play review ID |
| reviewer_name | TEXT | Name of the reviewer |
| rating | INTEGER | Star rating (1–5) |
| title | TEXT | Review title |
| body | TEXT | Review body text |
| review_date | TEXT | YYYY-MM-DD |
| thumbs_up | INTEGER | Helpful-vote count |
| app_version | TEXT | App version reviewed on |
| scraped_at | TEXT | Timestamp record was scraped |
| review_hash | TEXT UNIQUE | MD5 hash (dedup fallback) |

Feature columns are **not** persisted to SQLite — they are computed on demand and exported to CSV / markdown. This keeps the database compact and lets the feature pipeline evolve without schema migrations.

---

## Output Files

| File | Location | Description |
|---|---|---|
| Database | `reviews.db` | SQLite, all scraped reviews across runs |
| Raw CSV | `exports/{app}_{ts}.csv` | Cleaned reviews (compatible with `--from-csv`) |
| Features CSV | `exports/{app}_features_{ts}.csv` | Reviews + all feature columns except the 384-dim embedding |
| Summary report | `reports/{app}_{ts}.md` | Human-readable markdown analysis |
| Run log | `logs/pipeline_{ts}.log` | Per-run execution log |

---

## Dependencies

```
requests
beautifulsoup4
google-play-scraper
textblob
spacy
sentence-transformers
transformers
scikit-learn
numpy
```

Plus the spaCy English model: `python -m spacy download en_core_web_sm`.
