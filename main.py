from scraper.amazon_scraper import scrape_reviews

ASIN = "B0CFPJYX7P"

if __name__ == "__main__":
    reviews = scrape_reviews(ASIN, max_pages=5)
    print(f"\nDone. Total reviews scraped: {len(reviews)}")

    # Print first review to verify
    if reviews:
        print("\nSample review:")
        for key, value in reviews[0].items():
            print(f"  {key}: {value}")
