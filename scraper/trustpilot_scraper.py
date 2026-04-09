import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

BASE_URL = "https://www.trustpilot.com/review/{company}?page={page}"


def get_reviews_from_page(company, page_number):
    url = BASE_URL.format(company=company, page=page_number)
    print(f"Fetching: {url}")

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch page {page_number}: status {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    all_cards = soup.find_all("article", attrs={"data-service-review-card-paper": "true"})

    # Exclude carousel cards at the top — they repeat on every page
    review_elements = [
        card for card in all_cards
        if not any("carouselReviewCard" in c for c in card.get("class", []))
    ]

    if not review_elements:
        print(f"No reviews found on page {page_number}")
        return []

    reviews = []
    for element in review_elements:
        review = parse_review(element)
        if review:
            reviews.append(review)

    return reviews


def parse_review(element):
    try:
        # Rating from img alt e.g. "Rated 2 out of 5 stars"
        rating_img = element.find("img", class_=lambda c: c and "starRating" in c)
        rating = None
        if rating_img:
            alt = rating_img.get("alt", "")
            for word in alt.split():
                if word.isdigit():
                    rating = int(word)
                    break

        # Title
        title_element = element.find(attrs={"data-service-review-title-typography": True})

        # Body — non-carousel uses data-service-review-text-typography
        body_element = (
            element.find(attrs={"data-service-review-text-typography": True}) or
            element.find(attrs={"data-relevant-review-text-typography": True})
        )
        body = None
        if body_element:
            see_more = body_element.find("span")
            if see_more:
                see_more.decompose()
            body = body_element.text.strip()

        # Reviewer name
        name_element = element.find(attrs={"data-consumer-name-typography": True})

        # Date
        date_element = element.find("time")

        return {
            "reviewer_name": name_element.text.strip() if name_element else None,
            "rating": rating,
            "title": title_element.text.strip() if title_element else None,
            "body": body,
            "date": date_element.get("datetime") if date_element else None,
        }
    except Exception as e:
        print(f"Error parsing review: {e}")
        return None


def scrape_reviews(company, max_pages=5):
    all_reviews = []

    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        reviews = get_reviews_from_page(company, page)

        if not reviews:
            print("Stopping early — no more reviews found.")
            break

        all_reviews.extend(reviews)
        print(f"  Got {len(reviews)} reviews (total: {len(all_reviews)})")

        time.sleep(2)

    return all_reviews