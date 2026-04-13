from google_play_scraper import reviews, Sort
import time

def scrape_reviews(app_id, count=10000, lang="en", country="us"):
    all_reviews = []
    continuation_token = None
    batch_size = 200

    print(f"Scraping reviews for {app_id}...")

    while len(all_reviews) < count:
        try:
            batch, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=min(batch_size, count - len(all_reviews)),
                continuation_token=continuation_token
            )

            if not batch:
                print("No more reviews available.")
                break

            all_reviews.extend(batch)
            print(f"  Fetched {len(all_reviews)} reviews so far...")

            if not continuation_token:
                print("Reached end of reviews.")
                break

            time.sleep(1)

        except Exception as e:
            print(f"Error fetching reviews: {e}")
            break

    print(f"Done. Total fetched: {len(all_reviews)}")
    return all_reviews


def parse_reviews(raw_reviews):
    parsed = []

    for r in raw_reviews:
        parsed.append({
            "review_id":     r.get("reviewId"),
            "reviewer_name": r.get("userName"),
            "rating":        r.get("score"),
            "title":         None,
            "body":          r.get("content"),
            "date":          r.get("at").isoformat() if r.get("at") else None,
            "thumbs_up":     r.get("thumbsUpCount", 0),
            "app_version":   r.get("appVersion"),
        })

    return parsed