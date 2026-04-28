import csv
import os
from datetime import datetime, timezone

EXPORT_DIR = "exports"


def load_cleaned_csv(path):
    """Load a cleaned review export into the dict format used by the pipeline.

    Tolerant of older CSVs that pre-date the review_id / thumbs_up / app_version
    columns: missing columns simply become None / 0.
    """
    reviews = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = row.get("rating")
            try:
                rating = int(rating) if rating not in (None, "") else None
            except ValueError:
                rating = None

            thumbs_up = row.get("thumbs_up")
            try:
                thumbs_up = int(thumbs_up) if thumbs_up not in (None, "") else 0
            except ValueError:
                thumbs_up = 0

            reviews.append({
                "review_id":     row.get("review_id") or None,
                "reviewer_name": row.get("reviewer_name") or None,
                "rating":        rating,
                "title":         row.get("title") or None,
                "body":          row.get("body") or None,
                "date":          row.get("date") or None,
                "thumbs_up":     thumbs_up,
                "app_version":   row.get("app_version") or None,
            })
    print(f"Loaded {len(reviews)} reviews from {path}")
    return reviews


def export_to_csv(reviews, company_slug, exclude=()):
    """Export reviews to a timestamped CSV file, optionally dropping columns."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(EXPORT_DIR, f"{company_slug}_{timestamp}.csv")

    if not reviews:
        print("No reviews to export.")
        return

    exclude = set(exclude)
    fieldnames = [k for k in reviews[0].keys() if k not in exclude]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Exported {len(reviews)} reviews to {filename}")
    return filename