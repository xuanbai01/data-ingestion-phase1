import argparse
from scraper.play_scraper import scrape_reviews, parse_reviews
from database.db import create_tables, get_or_create_company, insert_reviews, verify_data
from pipeline.cleaner import clean_reviews
from pipeline.exporter import export_to_csv
from pipeline.logger import setup_logger

# App package names from Google Play
APPS = {
    "amazon":  ("com.amazon.mShop.android.shopping", "Amazon"),
    "ebay":    ("com.ebay.mobile", "eBay"),
    "walmart": ("com.walmart.android", "Walmart"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Google Play Review Scraper Pipeline")
    parser.add_argument("--apps", nargs="+", default=["amazon"],
                        choices=list(APPS.keys()),
                        help="One or more apps to scrape")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of reviews to fetch per app")
    parser.add_argument("--export", action="store_true",
                        help="Export results to CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()

    logger.info("Setting up database...")
    create_tables()

    for app_key in args.apps:
        app_id, app_name = APPS[app_key]
        logger.info(f"Starting pipeline for {app_name} ({app_id})...")

        # Scrape
        raw = scrape_reviews(app_id, count=args.count)
        parsed = parse_reviews(raw)
        logger.info(f"Scraped {len(parsed)} reviews.")

        # Clean
        logger.info("Cleaning reviews...")
        cleaned = clean_reviews(parsed)

        # Store
        logger.info("Saving to database...")
        company_id = get_or_create_company(app_name, app_id)
        insert_reviews(company_id, cleaned)

        # Export
        if args.export:
            logger.info("Exporting to CSV...")
            export_to_csv(cleaned, app_key)

        logger.info(f"Finished {app_name}.")

    verify_data()
    logger.info("Pipeline complete.")