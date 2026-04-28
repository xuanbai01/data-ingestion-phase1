import sqlite3
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

DB_PATH = "reviews.db"

# SQLite caps host-parameter count per statement (default 999, 32766 in newer
# builds). We chunk bulk SELECTs to stay safely under the lower limit.
_PARAM_CHUNK = 500


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

        -- Content-addressed feature cache. Decoupled from `reviews` (no FK)
        -- so caching also works in --from-csv mode where reviews aren't
        -- inserted into the DB at all. A row is a cache hit only when both
        -- the schema_version and embedder_model match the current pipeline.
        CREATE TABLE IF NOT EXISTS features (
            review_hash     TEXT PRIMARY KEY,
            schema_version  INTEGER NOT NULL,
            embedder_model  TEXT NOT NULL,
            polarity        REAL,
            subjectivity    REAL,
            aspects         TEXT,    -- JSON list[str]
            entities        TEXT,    -- JSON list[{text,label}]
            emotion         TEXT,
            urgency         REAL,
            embedding       BLOB,    -- numpy float32 bytes
            embedding_dim   INTEGER,
            created_at      TEXT NOT NULL
        );

        -- Per-run cluster snapshots. One row per cluster per pipeline run,
        -- keyed by app_slug (so it works in --from-csv mode without needing
        -- a companies row). Used by the time-dimension report to match
        -- issues across runs (Jaccard on aspect_set with centroid fallback)
        -- and to render sparklines / run-deltas.
        CREATE TABLE IF NOT EXISTS issue_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            app_slug        TEXT NOT NULL,
            run_id          TEXT NOT NULL,    -- ISO timestamp, one per pipeline run
            cluster_id      INTEGER NOT NULL, -- contiguous id within this run
            cluster_label   TEXT,             -- joined distinctive aspects, e.g. "delivery, package"
            aspect_set      TEXT NOT NULL,    -- JSON list[str], top-K distinctive aspects
            centroid        BLOB,             -- numpy float32 bytes (embedding centroid)
            centroid_dim    INTEGER,
            review_count    INTEGER NOT NULL,
            avg_rating      REAL,
            avg_polarity    REAL,
            avg_urgency     REAL,
            priority_score  REAL,
            is_issue        INTEGER NOT NULL, -- 1 if cluster qualified as a priority issue
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_app_run
            ON issue_snapshots(app_slug, run_id);

        -- LLM-generated cluster label cache (Phase V).
        -- Keyed by md5(sorted(review_hashes) + sorted(aspect_set)) so that
        -- a re-run on the same cluster contents reuses the label without
        -- another API call. Independent of app_slug since the same cluster
        -- can recur across apps (rare but cheap to share when it does).
        CREATE TABLE IF NOT EXISTS issue_labels (
            cache_key   TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            model       TEXT NOT NULL,
            created_at  TEXT NOT NULL
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
            (name, slug, datetime.now(timezone.utc).isoformat())
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

    scraped_at = datetime.now(timezone.utc).isoformat()
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


# ---------------------------------------------------------------------------
# Feature cache
#
# Content-addressed cache keyed by the same MD5 hash already used for review
# dedup. Lookups are gated on (schema_version, embedder_model): a row written
# under a different schema or with a different embedder is treated as a miss.
# ---------------------------------------------------------------------------

def compute_cache_key(review):
    """Stable cache key for a review dict (pre- or post-cleaning)."""
    return hash_review(
        review.get("reviewer_name"),
        review.get("date"),
        review.get("body"),
    )


def _encode_embedding(emb):
    """Encode a list / ndarray to (blob, dim). Returns (None, None) for None."""
    if emb is None:
        return None, None
    arr = np.asarray(emb, dtype=np.float32)
    return arr.tobytes(), int(arr.shape[0])


def _decode_embedding(blob, dim):
    """Decode a blob back to a Python list of floats. Returns None on shape mismatch."""
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim and arr.shape[0] != dim:
        return None
    return arr.tolist()


def load_features_batch(cache_keys, schema_version, embedder_model):
    """
    Bulk-load cached features for the given keys. Returns a dict mapping
    cache_key -> feature dict. Keys not present in the cache (or stored under
    a different schema_version / embedder_model) are silently absent.
    """
    if not cache_keys:
        return {}

    conn = get_connection()
    cursor = conn.cursor()

    out = {}
    for i in range(0, len(cache_keys), _PARAM_CHUNK):
        chunk = cache_keys[i:i + _PARAM_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(
            f"""
            SELECT review_hash, polarity, subjectivity, aspects, entities,
                   emotion, urgency, embedding, embedding_dim, embedder_model
            FROM features
            WHERE schema_version = ? AND review_hash IN ({placeholders})
            """,
            [schema_version, *chunk],
        )
        for row in cursor.fetchall():
            # Belt-and-suspenders: if someone swaps the embedder without
            # bumping schema_version, refuse to serve the stale embedding.
            if row["embedder_model"] != embedder_model:
                continue
            out[row["review_hash"]] = {
                "polarity":     row["polarity"],
                "subjectivity": row["subjectivity"],
                "aspects":      json.loads(row["aspects"]) if row["aspects"] else [],
                "entities":     json.loads(row["entities"]) if row["entities"] else [],
                "emotion":      row["emotion"],
                "urgency":      row["urgency"],
                "embedding":    _decode_embedding(row["embedding"], row["embedding_dim"]),
            }

    conn.close()
    return out


def save_features_batch(items, schema_version, embedder_model):
    """
    Bulk-write cached features. `items` is an iterable of (cache_key, feature_dict)
    pairs where feature_dict has the standard keys: polarity, subjectivity,
    aspects, entities, emotion, urgency, embedding. Existing rows with the same
    review_hash are replaced.
    """
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for key, f in items:
        emb_blob, emb_dim = _encode_embedding(f.get("embedding"))
        rows.append((
            key,
            schema_version,
            embedder_model,
            f.get("polarity"),
            f.get("subjectivity"),
            json.dumps(f.get("aspects") or []),
            json.dumps(f.get("entities") or []),
            f.get("emotion"),
            f.get("urgency"),
            emb_blob,
            emb_dim,
            now,
        ))

    if not rows:
        return

    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR REPLACE INTO features
            (review_hash, schema_version, embedder_model, polarity, subjectivity,
             aspects, entities, emotion, urgency, embedding, embedding_dim, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def clear_feature_cache(schema_version=None):
    """
    Clear cached features. With no argument, clears everything. With a version,
    keeps rows at that version and clears all others (useful after a bump).
    """
    conn = get_connection()
    cursor = conn.cursor()
    if schema_version is None:
        cursor.execute("DELETE FROM features")
    else:
        cursor.execute("DELETE FROM features WHERE schema_version != ?", (schema_version,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Cleared {deleted} cached feature rows.")
    return deleted


def feature_cache_stats():
    """Return basic stats about the feature cache. Useful for the verify step."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS c FROM features")
    total = cursor.fetchone()["c"]
    cursor.execute("""
        SELECT schema_version, embedder_model, COUNT(*) AS c
        FROM features
        GROUP BY schema_version, embedder_model
    """)
    breakdown = [(row["schema_version"], row["embedder_model"], row["c"]) for row in cursor.fetchall()]
    conn.close()
    return {"total": total, "by_version": breakdown}


# ---------------------------------------------------------------------------
# Issue snapshots (Phase IV — time dimension)
#
# One row per cluster per run. Stored by app_slug rather than company_id so
# snapshotting works in --from-csv mode without requiring a companies row.
# ---------------------------------------------------------------------------

def save_issue_snapshots(app_slug, run_id, snapshots):
    """
    Persist a run's clusters. `snapshots` is an iterable of dicts with keys:
        cluster_id, cluster_label, aspect_set (list[str]), centroid (list[float] | None),
        review_count, avg_rating, avg_polarity, avg_urgency, priority_score, is_issue (bool).

    Idempotent on (app_slug, run_id): a re-save with the same run_id wipes the
    prior rows for that run first, so a re-run doesn't double-count.
    """
    if not snapshots:
        return

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM issue_snapshots WHERE app_slug = ? AND run_id = ?",
        (app_slug, run_id),
    )

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for snap in snapshots:
        centroid_blob, centroid_dim = _encode_embedding(snap.get("centroid"))
        rows.append((
            app_slug,
            run_id,
            int(snap["cluster_id"]),
            snap.get("cluster_label"),
            json.dumps(list(snap.get("aspect_set") or [])),
            centroid_blob,
            centroid_dim,
            int(snap["review_count"]),
            snap.get("avg_rating"),
            snap.get("avg_polarity"),
            snap.get("avg_urgency"),
            snap.get("priority_score"),
            1 if snap.get("is_issue") else 0,
            now,
        ))

    cursor.executemany(
        """
        INSERT INTO issue_snapshots
            (app_slug, run_id, cluster_id, cluster_label, aspect_set,
             centroid, centroid_dim, review_count, avg_rating, avg_polarity,
             avg_urgency, priority_score, is_issue, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()
    conn.close()


def _row_to_snapshot(row):
    return {
        "app_slug":       row["app_slug"],
        "run_id":         row["run_id"],
        "cluster_id":     row["cluster_id"],
        "cluster_label":  row["cluster_label"],
        "aspect_set":     json.loads(row["aspect_set"]) if row["aspect_set"] else [],
        "centroid":       _decode_embedding(row["centroid"], row["centroid_dim"]),
        "review_count":   row["review_count"],
        "avg_rating":     row["avg_rating"],
        "avg_polarity":   row["avg_polarity"],
        "avg_urgency":    row["avg_urgency"],
        "priority_score": row["priority_score"],
        "is_issue":       bool(row["is_issue"]),
        "created_at":     row["created_at"],
    }


def recent_run_ids(app_slug, limit=12):
    """Return the most recent run_ids for an app, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT run_id FROM issue_snapshots
        WHERE app_slug = ?
        GROUP BY run_id
        ORDER BY run_id DESC
        LIMIT ?
        """,
        (app_slug, limit),
    )
    out = [row["run_id"] for row in cursor.fetchall()]
    conn.close()
    return out


def load_snapshots(app_slug, run_ids):
    """Bulk-load snapshots for a set of (app_slug, run_id) pairs.

    Returns a dict mapping run_id -> list[snapshot dict] in cluster_id order.
    Empty if no run_ids supplied.
    """
    if not run_ids:
        return {}

    conn = get_connection()
    cursor = conn.cursor()

    out = defaultdict(list)
    for i in range(0, len(run_ids), _PARAM_CHUNK):
        chunk = run_ids[i:i + _PARAM_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(
            f"""
            SELECT * FROM issue_snapshots
            WHERE app_slug = ? AND run_id IN ({placeholders})
            ORDER BY run_id DESC, cluster_id ASC
            """,
            [app_slug, *chunk],
        )
        for row in cursor.fetchall():
            out[row["run_id"]].append(_row_to_snapshot(row))

    conn.close()
    return dict(out)


def load_prior_run_snapshots(app_slug, before_run_id):
    """Return (prior_run_id, [snapshots]) for the most recent run strictly
    before `before_run_id`. Returns (None, []) when there is no prior run.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT run_id FROM issue_snapshots
        WHERE app_slug = ? AND run_id < ?
        GROUP BY run_id
        ORDER BY run_id DESC
        LIMIT 1
        """,
        (app_slug, before_run_id),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None, []
    prior_run_id = row["run_id"]
    return prior_run_id, load_snapshots(app_slug, [prior_run_id]).get(prior_run_id, [])


# ---------------------------------------------------------------------------
# LLM cluster label cache (Phase V)
# ---------------------------------------------------------------------------

def load_issue_label(cache_key):
    """Return cached label string for the given cache_key, or None on miss."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT label FROM issue_labels WHERE cache_key = ?",
        (cache_key,),
    )
    row = cursor.fetchone()
    conn.close()
    return row["label"] if row else None


def save_issue_label(cache_key, label, model):
    """Insert or replace a label cache row."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO issue_labels
            (cache_key, label, model, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (cache_key, label, model, now),
    )
    conn.commit()
    conn.close()