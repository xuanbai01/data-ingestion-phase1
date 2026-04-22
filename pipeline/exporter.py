import csv
import os
from datetime import datetime

EXPORT_DIR = "exports"


def load_cleaned_csv(path):
    """Load a cleaned review export into the dict format used by the pipeline."""
    reviews = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = row.get("rating")
            try:
                rating = int(rating) if rating not in (None, "") else None
            except ValueError:
                rating = None
            reviews.append({
                "reviewer_name": row.get("reviewer_name") or None,
                "rating": rating,
                "title": row.get("title") or None,
                "body": row.get("body") or None,
                "date": row.get("date") or None,
            })
    print(f"Loaded {len(reviews)} reviews from {path}")
    return reviews


def export_to_csv(reviews, company_slug):
    """Export reviews to a timestamped CSV file."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(EXPORT_DIR, f"{company_slug}_{timestamp}.csv")

    if not reviews:
        print("No reviews to export.")
        return

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Exported {len(reviews)} reviews to {filename}")
    return filename