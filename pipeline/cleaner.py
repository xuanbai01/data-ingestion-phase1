from datetime import datetime


def clean_reviews(reviews):
    """Clean and normalize a list of raw review dicts."""
    cleaned = []

    for review in reviews:
        cleaned_review = clean_review(review)
        if cleaned_review:
            cleaned.append(cleaned_review)

    missing_body = sum(1 for r in cleaned if not r["body"])
    print(f"Cleaned {len(cleaned)} reviews ({missing_body} missing body text).")
    return cleaned


def clean_review(review):
    """Clean and normalize a single review dict."""
    try:
        return {
            "review_id":     clean_text(review.get("review_id")),
            "reviewer_name": clean_text(review.get("reviewer_name")),
            "rating":        clean_rating(review.get("rating")),
            "title":         clean_text(review.get("title")),
            "body":          clean_text(review.get("body")),
            "date":          clean_date(review.get("date")),
            "thumbs_up":     clean_thumbs_up(review.get("thumbs_up")),
            "app_version":   clean_text(review.get("app_version")),
        }
    except Exception as e:
        print(f"Error cleaning review: {e}")
        return None


def clean_thumbs_up(value):
    """Coerce thumbs_up to a non-negative int; 0 if missing or invalid."""
    if value is None or value == "":
        return 0
    try:
        n = int(value)
        return n if n >= 0 else 0
    except (ValueError, TypeError):
        return 0


def clean_text(value):
    """Strip whitespace and return None if empty."""
    if not value:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def clean_rating(value):
    """Ensure rating is an integer between 1 and 5."""
    if value is None:
        return None
    try:
        rating = int(value)
        if 1 <= rating <= 5:
            return rating
        return None  # invalid range
    except (ValueError, TypeError):
        return None


def clean_date(value):
    """Normalize ISO date string to YYYY-MM-DD format."""
    if not value:
        return None
    try:
        # Trustpilot dates: "2026-04-09T02:25:46.000Z"
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None