import argparse
from scraper.trustpilot_scraper import scrape_reviews
from database.db import create_tables, get_or_create_company, insert_reviews, verify_data
from pipeline.cleaner import clean_reviews
from pipeline.exporter import export_to_csv
from pipeline.logger import setup_logger


# Add more companies here anytime
COMPANIES = {
    "www.amazon.com":  "Amazon",
    "www.ebay.com":    "eBay",
    "www.walmart.com": "Walmart",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Trustpilot Review Scraper Pipeline")
    parser.add_argument(
        "--companies",
        nargs="+",
        default=["www.amazon.com"],
        choices=list(COMPANIES.keys()),
        help="One or more company slugs to scrape"
    )
    parser.add_argument("--pages", type=int, default=5,
                        help="Number of pages to scrape per company")
    parser.add_argument("--export", action="store_true",
                        help="Export results to CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()

    logger.info("Setting up database...")
    create_tables()

    for slug in args.companies:
        name = COMPANIES[slug]
        logger.info(f"Starting pipeline for {name} ({slug})...")

        # Scrape
        logger.info(f"Scraping {args.pages} pages...")
        raw_reviews = scrape_reviews(slug, max_pages=args.pages)
        logger.info(f"Scraped {len(raw_reviews)} raw reviews.")

        # Clean
        logger.info("Cleaning reviews...")
        cleaned = clean_reviews(raw_reviews)

        # Store
        logger.info("Saving to database...")
        company_id = get_or_create_company(name, slug)
        insert_reviews(company_id, cleaned)

        # Export
        if args.export:
            logger.info("Exporting to CSV...")
            export_to_csv(cleaned, slug)

        logger.info(f"Finished {name}.")

    # Final DB stats
    verify_data()
    logger.info("Pipeline complete.")