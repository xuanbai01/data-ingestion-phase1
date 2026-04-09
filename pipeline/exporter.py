import csv
import os
from datetime import datetime

EXPORT_DIR = "exports"


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