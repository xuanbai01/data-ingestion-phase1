import argparse
from scraper.play_scraper import scrape_reviews, parse_reviews
from database.db import create_tables, get_or_create_company, insert_reviews, verify_data
from pipeline.cleaner import clean_reviews
from pipeline.exporter import export_to_csv, load_cleaned_csv
from pipeline.logger import setup_logger
from pipeline.feature_engineering import run_pipeline, validate_features
from pipeline.summarizer import generate_report


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
                        help="One or more apps to process")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of reviews to fetch per app (ignored with --from-csv)")
    parser.add_argument("--export", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Path to a cleaned CSV export. Skips scraping/DB and runs feature engineering on that file instead.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip the feature cache; recompute every feature from scratch.")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Force a fixed number of clusters (default: adaptive K via silhouette score).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()

    # Always set up the DB — even in --from-csv mode the feature cache lives
    # there. create_tables uses CREATE TABLE IF NOT EXISTS so this is idempotent.
    logger.info("Setting up database...")
    create_tables()

    for app_key in args.apps:
        app_id, app_name = APPS[app_key]
        brand_stopwords = [app_name.lower(), app_key.lower()]

        if args.from_csv:
            logger.info(f"Loading reviews for {app_name} from {args.from_csv}...")
            cleaned = load_cleaned_csv(args.from_csv)
        else:
            logger.info(f"Starting pipeline for {app_name} ({app_id})...")

            raw = scrape_reviews(app_id, count=args.count)
            parsed = parse_reviews(raw)
            logger.info(f"Scraped {len(parsed)} reviews.")

            logger.info("Cleaning reviews...")
            cleaned = clean_reviews(parsed)

            logger.info("Saving to database...")
            company_id = get_or_create_company(app_name, app_id)
            insert_reviews(company_id, cleaned)

        # Feature engineering
        logger.info("Running feature engineering...")
        featured = run_pipeline(
            cleaned,
            n_clusters=args.n_clusters,
            brand_stopwords=brand_stopwords,
            use_cache=not args.no_cache,
        )
        validate_features(featured)

        # Summary report. Pass app_id as the snapshot slug so the report
        # can persist per-cluster snapshots and render a Run Delta section
        # against this app's prior runs.
        logger.info("Generating summary report...")
        generate_report(featured, app_name, app_slug=app_id)

        # Export
        if args.export:
            logger.info("Exporting to CSV...")
            if not args.from_csv:
                export_to_csv(cleaned, app_key)
            # Drop embedding column — 768 floats per row makes the CSV huge
            # and unreadable, and clustering has already consumed it.
            export_to_csv(featured, f"{app_key}_features", exclude=["embedding"])

        logger.info(f"Finished {app_name}.")

    if not args.from_csv:
        verify_data()
    logger.info("Pipeline complete.")
