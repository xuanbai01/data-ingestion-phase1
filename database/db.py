import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "reviews.db"


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    return conn


def hash_review(reviewer_name, review_date, body):
    """Generate a unique hash for a review."""
    raw = f"{reviewer_name}|{review_date}|{body or ''}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS companies (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            slug        TEXT NOT NULL UNIQUE,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id      INTEGER NOT NULL,
            review_id       TEXT UNIQUE,
            reviewer_name   TEXT,
            rating          INTEGER,
            title           TEXT,
            body            TEXT,
            review_date     TEXT,
            thumbs_up       INTEGER DEFAULT 0,
            app_version     TEXT,
            scraped_at      TEXT NOT NULL,
            review_hash     TEXT UNIQUE,
            FOREIGN KEY (company_id) REFERENCES companies(id)
        );
    """)

    conn.commit()
    conn.close()
    print("Tables created successfully.")


def get_or_create_company(name, slug):
    """Insert company if it doesn't exist, return its id."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM companies WHERE slug = ?", (slug,))
    row = cursor.fetchone()

    if row:
        company_id = row["id"]
    else:
        cursor.execute(
            "INSERT INTO companies (name, slug, created_at) VALUES (?, ?, ?)",
            (name, slug, datetime.utcnow().isoformat())
        )
        conn.commit()
        company_id = cursor.lastrowid
        print(f"Created company: {name} (id={company_id})")

    conn.close()
    return company_id


def review_exists(company_id, reviewer_name, review_date):
    """Check if a review already exists in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id FROM reviews 
        WHERE company_id = ? AND reviewer_name = ? AND review_date = ?
    """, (company_id, reviewer_name, review_date))

    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def insert_reviews(company_id, reviews):
    conn = get_connection()
    cursor = conn.cursor()

    scraped_at = datetime.utcnow().isoformat()
    inserted = 0
    skipped = 0

    for review in reviews:
        # Use review_id if available, otherwise fall back to hash
        review_id = review.get("review_id")
        review_hash = hash_review(
            review.get("reviewer_name"),
            review.get("date"),
            review.get("body")
        )

        # Skip if review_id or hash already exists
        if review_id:
            cursor.execute("SELECT id FROM reviews WHERE review_id = ?", (review_id,))
        else:
            cursor.execute("SELECT id FROM reviews WHERE review_hash = ?", (review_hash,))

        if cursor.fetchone():
            skipped += 1
            continue

        cursor.execute("""
            INSERT INTO reviews 
                (company_id, review_id, reviewer_name, rating, title, body,
                 review_date, thumbs_up, app_version, scraped_at, review_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            company_id,
            review_id,
            review.get("reviewer_name"),
            review.get("rating"),
            review.get("title"),
            review.get("body"),
            review.get("date"),
            review.get("thumbs_up", 0),
            review.get("app_version"),
            scraped_at,
            review_hash
        ))
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Inserted {inserted} reviews, skipped {skipped} duplicates.")

def verify_data():
    """Print basic stats to confirm data loaded correctly."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM reviews")
    total = cursor.fetchone()["count"]

    cursor.execute("SELECT AVG(rating) as avg FROM reviews WHERE rating IS NOT NULL")
    avg_rating = cursor.fetchone()["avg"]

    cursor.execute("SELECT COUNT(*) as count FROM reviews WHERE body IS NULL")
    missing_body = cursor.fetchone()["count"]

    print(f"\n--- Database Verification ---")
    print(f"Total reviews:     {total}")
    print(f"Average rating:    {avg_rating:.2f}" if avg_rating else "Average rating: N/A")
    print(f"Missing body:      {missing_body}")

    conn.close()

def print_sample(limit=5):
    """Print a sample of reviews from the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT r.reviewer_name, r.rating, r.title, r.body, r.review_date
        FROM reviews r
        ORDER BY r.id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    print(f"\n--- Last {limit} Reviews ---")
    for row in rows:
        print(f"\nReviewer: {row['reviewer_name']}")
        print(f"Rating:   {row['rating']}/5")
        print(f"Title:    {row['title']}")
        print(f"Body:     {row['body']}")
        print(f"Date:     {row['review_date']}")
        print("-" * 40)