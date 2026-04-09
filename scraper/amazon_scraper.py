from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

REVIEWS_URL = "https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&pageNumber={page}&sortBy=recent"


def create_driver():
    """Launch a Chrome browser that looks like a real user."""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    # Mask webdriver fingerprint
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver


def get_reviews_from_page(driver, asin, page_number):
    """Fetch and parse reviews from a single page."""
    url = REVIEWS_URL.format(asin=asin, page=page_number)
    print(f"Fetching: {url}")

    driver.get(url)

    # Wait for reviews to load or detect login wall
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-hook="review"]'))
        )
    except:
        page_title = driver.title
        print(f"Reviews not found — page title: {page_title}")
        return []

    # Let the page fully settle
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    review_elements = soup.find_all("div", attrs={"data-hook": "review"})

    if not review_elements:
        print(f"No reviews parsed on page {page_number}")
        return []

    reviews = []
    for element in review_elements:
        review = parse_review(element)
        if review:
            reviews.append(review)

    return reviews


def parse_review(element):
    """Extract fields from a single review element."""
    try:
        rating_element = element.find(attrs={"data-hook": "review-star-rating"})
        title_element = element.find(attrs={"data-hook": "review-title"})
        body_element = element.find(attrs={"data-hook": "review-body"})
        date_element = element.find(attrs={"data-hook": "review-date"})
        name_element = element.find(class_="a-profile-name")

        rating_text = rating_element.text.strip() if rating_element else None
        rating = int(float(rating_text.split(" ")[0])) if rating_text else None

        return {
            "reviewer_name": name_element.text.strip() if name_element else None,
            "rating": rating,
            "title": title_element.text.strip() if title_element else None,
            "body": body_element.text.strip() if body_element else None,
            "date": date_element.text.strip() if date_element else None,
        }
    except Exception as e:
        print(f"Error parsing review: {e}")
        return None


def scrape_reviews(asin, max_pages=5):
    """Scrape multiple pages of reviews for a given product."""
    driver = create_driver()
    all_reviews = []

    try:
        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}...")
            reviews = get_reviews_from_page(driver, asin, page)

            if not reviews:
                print("Stopping early — no more reviews found.")
                break

            all_reviews.extend(reviews)
            print(f"  Got {len(reviews)} reviews (total: {len(all_reviews)})")

            time.sleep(3)

    finally:
        driver.quit()  # always close the browser

    return all_reviews