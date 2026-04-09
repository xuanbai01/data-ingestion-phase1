# Data Ingestion System — Phase I

A modular, automated pipeline for scraping, cleaning, and storing user reviews from Trustpilot into a relational database for downstream sentiment analysis.

---

## Project Structure

```
data_ingestion/
├── scraper/             # Data acquisition layer
│   └── trustpilot_scraper.py
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

### Scrape a single company
```bash
python main.py --companies www.amazon.com --pages 10 --export
```

### Scrape multiple companies
```bash
python main.py --companies www.amazon.com www.ebay.com www.walmart.com --pages 10 --export
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--companies` | `www.amazon.com` | One or more Trustpilot company slugs |
| `--pages` | `5` | Number of review pages to scrape per company |
| `--export` | `False` | Export results to CSV in `/exports` |

> **Note:** Trustpilot limits unauthenticated access to 10 pages (200 reviews) per company.

---

## Pipeline Overview

```
Scrape (Trustpilot) → Clean & Normalize → Store (SQLite) → Export (CSV)
```

1. **Scraper** — fetches paginated reviews using Requests and BeautifulSoup
2. **Cleaner** — strips whitespace, normalizes dates to YYYY-MM-DD, validates ratings
3. **Database** — stores in a relational schema with duplicate prevention via MD5 hash
4. **Exporter** — saves timestamped CSV backups to `/exports` for downstream use
5. **Logger** — writes timestamped logs to `/logs` for every pipeline run

---

## Database Schema

### `companies`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| name | TEXT | Company display name |
| slug | TEXT | Trustpilot URL slug (unique) |
| created_at | TEXT | Record creation timestamp |

### `reviews`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| company_id | INTEGER | Foreign key → companies.id |
| reviewer_name | TEXT | Name of the reviewer |
| rating | INTEGER | Star rating (1–5) |
| title | TEXT | Review title |
| body | TEXT | Review body text |
| review_date | TEXT | Date of review (YYYY-MM-DD) |
| scraped_at | TEXT | Timestamp when record was scraped |
| review_hash | TEXT | MD5 hash for deduplication (unique) |

---

## Adding a New Company

Add the company slug to the `COMPANIES` dict in `main.py`:

```python
COMPANIES = {
    "www.amazon.com":  "Amazon",
    "www.ebay.com":    "eBay",
    "www.walmart.com": "Walmart",
    "www.yourcompany.com": "Your Company",  # add here
}
```

Then run:

```bash
python main.py --companies www.yourcompany.com --pages 10 --export
```

---

## Output Files

| File | Location | Description |
|---|---|---|
| Database | `reviews.db` | SQLite database with all scraped reviews |
| CSV export | `exports/{slug}_{timestamp}.csv` | Per-run CSV backup |
| Log file | `logs/pipeline_{timestamp}.log` | Per-run execution log |

---

## Requirements

```
requests
beautifulsoup4
```
