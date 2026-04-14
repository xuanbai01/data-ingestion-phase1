# Data Ingestion System — Phase I

A modular, automated pipeline for scraping, cleaning, and storing app reviews from Google Play Store into a relational database for downstream sentiment analysis.

---

## Project Structure

```
data_ingestion/
├── scraper/             # Data acquisition layer
│   ├── play_scraper.py
│   └── amazon_scraper.py  # archived - Amazon blocks unauthenticated scraping
├── database/            # Storage layer
│   └── db.py
├── pipeline/            # Transformation layer
│   ├── cleaner.py
│   ├── exporter.py
│   └── logger.py
├── logs/                # Auto-generated run logs
├── exports/             # Auto-generated CSV exports
├── reviews.db           # SQLite database (auto-created)
├── main.py              # Pipeline entrypoint
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
```

---

## Usage

### Scrape a single app
```bash
python main.py --apps amazon --count 10000 --export
```

### Scrape multiple apps
```bash
python main.py --apps amazon ebay walmart --count 10000 --export
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--apps` | `amazon` | One or more app keys to scrape |
| `--count` | `10000` | Number of reviews to fetch per app |
| `--export` | `False` | Export results to CSV in `/exports` |

### Supported Apps

| Key | App | Package ID |
|---|---|---|
| `amazon` | Amazon Shopping | `com.amazon.mShop.android.shopping` |
| `ebay` | eBay | `com.ebay.mobile` |
| `walmart` | Walmart | `com.walmart.android` |

---

## Pipeline Overview

```
Scrape (Google Play) → Clean & Normalize → Store (SQLite) → Export (CSV)
```

1. **Scraper** — fetches reviews from Google Play Store using `google-play-scraper`, with pagination support via continuation tokens
2. **Cleaner** — strips whitespace, normalizes dates to YYYY-MM-DD, validates ratings
3. **Database** — stores in a relational schema with duplicate prevention via `review_id` and MD5 hash fallback
4. **Exporter** — saves timestamped CSV backups to `/exports` for downstream use
5. **Logger** — writes timestamped logs to `/logs` for every pipeline run

---

## Database Schema

### `companies`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| name | TEXT | App display name |
| slug | TEXT | Google Play package ID (unique) |
| created_at | TEXT | Record creation timestamp |

### `reviews`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| company_id | INTEGER | Foreign key → companies.id |
| review_id | TEXT | Google Play review ID (unique) |
| reviewer_name | TEXT | Name of the reviewer |
| rating | INTEGER | Star rating (1–5) |
| title | TEXT | Review title |
| body | TEXT | Review body text |
| review_date | TEXT | Date of review (YYYY-MM-DD) |
| thumbs_up | INTEGER | Number of helpful votes |
| app_version | TEXT | App version reviewed on |
| scraped_at | TEXT | Timestamp when record was scraped |
| review_hash | TEXT | MD5 hash for deduplication fallback |

---

## Adding a New App

Add the app to the `APPS` dict in `main.py`:

```python
APPS = {
    "amazon":  ("com.amazon.mShop.android.shopping", "Amazon"),
    "ebay":    ("com.ebay.mobile", "eBay"),
    "walmart": ("com.walmart.android", "Walmart"),
    "yourapp": ("com.your.app.package", "Your App"),  # add here
}
```

Then run:

```bash
python main.py --apps yourapp --count 10000 --export
```

---

## Output Files

| File | Location | Description |
|---|---|---|
| Database | `reviews.db` | SQLite database with all scraped reviews |
| CSV export | `exports/{app}_{timestamp}.csv` | Per-run CSV backup |
| Log file | `logs/pipeline_{timestamp}.log` | Per-run execution log |

---

## Requirements

```
requests
beautifulsoup4
google-play-scraper
```
