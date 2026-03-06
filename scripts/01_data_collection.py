"""
01_data_collection.py
=====================
Web scraping pipeline for U.S. real estate data.

Sources:
  1. Zillow.com     → Property listings, Zestimates, rent estimates
  2. Redfin.com     → Market trends, recently sold, price history
  3. Realtor.com    → Supplementary listings + neighborhood data
  4. Census.gov API → Median household income by ZIP

Usage:
  python scripts/01_data_collection.py --market "Austin, TX" --pages 1 --no-headless --sources redfin
  python scripts/01_data_collection.py --all --pages 2
"""

import os
import re
import csv
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)
import undetected_chromedriver as uc

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RAW_DIR / "scraping.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── Target Markets ───────────────────────────────────────────────────────────

TARGET_MARKETS = [
    {"city": "Austin",         "state": "TX", "zip_codes": ["78701","78702","78704","78745","78750"]},
    {"city": "Boise",          "state": "ID", "zip_codes": ["83702","83706","83709","83713","83716"]},
    {"city": "Raleigh",        "state": "NC", "zip_codes": ["27601","27603","27606","27609","27612"]},
    {"city": "Nashville",      "state": "TN", "zip_codes": ["37201","37203","37206","37209","37211"]},
    {"city": "Phoenix",        "state": "AZ", "zip_codes": ["85003","85004","85006","85008","85016"]},
    {"city": "Tampa",          "state": "FL", "zip_codes": ["33602","33606","33609","33611","33629"]},
    {"city": "Charlotte",      "state": "NC", "zip_codes": ["28202","28203","28205","28209","28210"]},
    {"city": "Salt Lake City", "state": "UT", "zip_codes": ["84101","84102","84103","84105","84106"]},
    {"city": "Denver",         "state": "CO", "zip_codes": ["80202","80205","80209","80210","80220"]},
    {"city": "Jacksonville",   "state": "FL", "zip_codes": ["32202","32204","32207","32210","32216"]},
    {"city": "New York",       "state": "NY", "zip_codes": ["10001","10002","10003","10011","10025"]},
    {"city": "Boston",         "state": "MA", "zip_codes": ["02108","02116","02118","02127","02130"]},
    {"city": "San Francisco",  "state": "CA", "zip_codes": ["94102","94103","94107","94110","94117"]},
    {"city": "Los Angeles",    "state": "CA", "zip_codes": ["90012","90015","90026","90036","90046"]},
    {"city": "Seattle",        "state": "WA", "zip_codes": ["98101","98103","98105","98109","98115"]},
    {"city": "Chicago",        "state": "IL", "zip_codes": ["60601","60607","60614","60625","60647"]},
    {"city": "Buffalo",        "state": "NY", "zip_codes": ["14201","14204","14207","14209","14213"]},
    {"city": "Hartford",       "state": "CT", "zip_codes": ["06103","06105","06106","06112","06114"]},
    {"city": "Durham",         "state": "NC", "zip_codes": ["27701","27703","27704","27705","27707"]},
    {"city": "St. Louis",      "state": "MO", "zip_codes": ["63101","63103","63104","63108","63118"]},
    ]


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class PropertyListing:
    source: str = ""
    scrape_timestamp: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    full_address: str = ""
    price: Optional[float] = None
    original_price: Optional[float] = None
    monthly_rent: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    sqft: Optional[int] = None
    lot_sqft: Optional[int] = None
    year_built: Optional[int] = None
    property_type: str = ""
    listing_status: str = ""
    days_on_market: Optional[int] = None
    price_per_sqft: Optional[float] = None
    hoa_fee: Optional[float] = None
    zestimate: Optional[float] = None
    rent_zestimate: Optional[float] = None
    tax_assessed_value: Optional[float] = None
    annual_tax: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    listing_url: str = ""
    mls_id: str = ""
    broker: str = ""
    description: str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def human_delay(min_sec=2, max_sec=5):
    time.sleep(random.uniform(min_sec, max_sec))


def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    text = re.sub(r'(From|Est\.|Estimated|Starting at|Zestimate:?)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\/mo(nth)?', '', text, flags=re.IGNORECASE)
    match = re.search(r'\$?([\d,]+\.?\d*)\s*(K|M)?', text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1).replace(',', ''))
    mult = match.group(2)
    if mult:
        if mult.upper() == 'K': value *= 1_000
        elif mult.upper() == 'M': value *= 1_000_000
    return value


def parse_int(text: str) -> Optional[int]:
    if not text: return None
    match = re.search(r'([\d,]+)', text)
    return int(match.group(1).replace(',', '')) if match else None


def parse_float(text: str) -> Optional[float]:
    if not text: return None
    match = re.search(r'([\d,]+\.?\d*)', text)
    return float(match.group(1).replace(',', '')) if match else None


def save_to_csv(records: list, filename: str):
    if not records:
        logger.warning(f"No records to save for {filename}")
        return
    filepath = RAW_DIR / filename
    fieldnames = list(asdict(records[0]).keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    logger.info(f"Saved {len(records)} records to {filepath}")


# ─── Browser Factory ──────────────────────────────────────────────────────────

class BrowserFactory:
    @staticmethod
    def create(headless: bool = True) -> webdriver.Chrome:
        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        width = random.randint(1280, 1440)
        height = random.randint(800, 900)
        options.add_argument(f"--window-size={width},{height}")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        driver = uc.Chrome(options=options, use_subprocess=True, version_main=145)
        driver.set_page_load_timeout(30)
        logger.info(f"Browser initialized | Size: {width}x{height}")
        return driver


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER 1: ZILLOW
# ═══════════════════════════════════════════════════════════════════════════════

class ZillowScraper:
    """
    Scrapes Zillow search results.
    Zillow is JS-rendered (React/Next.js) — requires Selenium.
    Extracts data from __NEXT_DATA__ JSON or falls back to HTML parsing.
    """

    BASE_URL = "https://www.zillow.com"

    def __init__(self, driver):
        self.driver = driver

    def _build_search_url(self, city: str, state: str, status: str = "forSale", page: int = 1) -> str:
        slug = f"{city.lower().replace(' ', '-')}-{state.lower()}"
        if status == "forRent":
            path = f"/{slug}/rentals/"
        elif status == "recentlySold":
            path = f"/{slug}/sold/"
        else:
            path = f"/{slug}/"
        if page > 1:
            path = path.rstrip("/") + f"/{page}_p/"
        return self.BASE_URL + path

    def _scroll_to_load(self, pause=1.5, scrolls=3):
        for i in range(scrolls):
            self.driver.execute_script(
                f"window.scrollTo(0, document.body.scrollHeight * {(i+1)/(scrolls+1)});"
            )
            time.sleep(pause + random.uniform(0, 0.5))
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    def _extract_next_data(self) -> dict:
        try:
            script = self.driver.find_element(By.ID, "__NEXT_DATA__")
            return json.loads(script.get_attribute("innerHTML"))
        except (NoSuchElementException, json.JSONDecodeError):
            return {}

    def _parse_from_json(self, data: dict, city: str, state: str) -> list:
        listings = []
        try:
            page_props = data.get("props", {}).get("pageProps", {})
            search_state = page_props.get("searchPageState", {})
            for cat_key in ["cat1", "cat2"]:
                cat = search_state.get(cat_key, {})
                list_results = cat.get("searchResults", {}).get("listResults", [])
                for item in list_results:
                    l = PropertyListing(source="zillow", scrape_timestamp=datetime.now().isoformat(), city=city, state=state)
                    l.full_address = item.get("address", "")
                    l.price = item.get("unformattedPrice") or item.get("price")
                    if isinstance(l.price, str): l.price = parse_price(l.price)
                    l.zestimate = item.get("zestimate")
                    l.bedrooms = item.get("beds")
                    l.bathrooms = item.get("baths")
                    l.sqft = item.get("area")
                    l.latitude = item.get("latLong", {}).get("latitude")
                    l.longitude = item.get("latLong", {}).get("longitude")
                    l.listing_status = item.get("statusType", "").lower().replace("_", " ")
                    l.listing_url = item.get("detailUrl", "")
                    if l.listing_url and not l.listing_url.startswith("http"):
                        l.listing_url = self.BASE_URL + l.listing_url
                    hdp = item.get("hdpData", {}).get("homeInfo", {})
                    if hdp:
                        l.zip_code = str(hdp.get("zipcode", ""))
                        l.property_type = hdp.get("homeType", "")
                        l.year_built = hdp.get("yearBuilt")
                        l.days_on_market = hdp.get("daysOnZillow")
                        l.rent_zestimate = hdp.get("rentZestimate")
                        l.tax_assessed_value = hdp.get("taxAssessedValue")
                    if l.price and l.sqft and l.sqft > 0:
                        l.price_per_sqft = round(l.price / l.sqft, 2)
                    listings.append(l)
            logger.info(f"Extracted {len(listings)} listings from Zillow JSON")
        except Exception as e:
            logger.error(f"Error parsing Zillow JSON: {e}")
        return listings

    def _parse_from_html(self, city: str, state: str) -> list:
        listings = []
        try:
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            selectors = [
                "article[data-test='property-card']",
                "div[class*='StyledPropertyCard']",
                "div.property-card",
            ]
            cards = []
            for sel in selectors:
                cards = soup.select(sel)
                if cards: break
            if not cards:
                cards = soup.find_all("article")

            for card in cards:
                l = PropertyListing(source="zillow_html", scrape_timestamp=datetime.now().isoformat(), city=city, state=state)
                price_el = card.select_one("[data-test='property-card-price']") or card.find("span", string=re.compile(r'\$[\d,]+'))
                if price_el: l.price = parse_price(price_el.get_text())
                addr_el = card.select_one("[data-test='property-card-addr']") or card.select_one("address")
                if addr_el:
                    l.full_address = addr_el.get_text(strip=True)
                    zip_m = re.search(r'(\d{5})$', l.full_address)
                    if zip_m: l.zip_code = zip_m.group(1)
                text = card.get_text(" ", strip=True)
                beds_m = re.search(r'(\d+)\s*(?:bd|bed|br)', text, re.IGNORECASE)
                baths_m = re.search(r'([\d.]+)\s*(?:ba|bath)', text, re.IGNORECASE)
                sqft_m = re.search(r'([\d,]+)\s*(?:sq\s*ft|sqft)', text, re.IGNORECASE)
                if beds_m: l.bedrooms = int(beds_m.group(1))
                if baths_m: l.bathrooms = parse_float(baths_m.group(1))
                if sqft_m: l.sqft = parse_int(sqft_m.group(1))
                link = card.select_one("a[href*='/homedetails/']") or card.find("a", href=True)
                if link:
                    href = link.get("href", "")
                    if not href.startswith("http"): href = self.BASE_URL + href
                    l.listing_url = href
                if l.price and l.sqft and l.sqft > 0:
                    l.price_per_sqft = round(l.price / l.sqft, 2)
                if l.price and l.price > 10000:
                    listings.append(l)
            logger.info(f"Extracted {len(listings)} listings from Zillow HTML")
        except Exception as e:
            logger.error(f"Error parsing Zillow HTML: {e}")
        return listings

    def scrape_market(self, city: str, state: str, max_pages: int = 3,
                      statuses=None, scrape_details: bool = False) -> list:
        if statuses is None:
            statuses = ["forSale", "forRent", "recentlySold"]
        all_listings = []
        for status in statuses:
            logger.info(f"\nScraping Zillow: {city}, {state} | {status}")
            for page in range(1, max_pages + 1):
                url = self._build_search_url(city, state, status, page)
                logger.info(f"Page {page}: {url}")
                try:
                    self.driver.get(url)
                    human_delay(3, 7)
                    try:
                        WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "article, [data-test='property-card']"))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout on page {page}")
                    if "captcha" in self.driver.page_source.lower() or "perimeter" in self.driver.current_url.lower():
                        logger.error("CAPTCHA detected! Pausing 60s — solve it in the browser.")
                        time.sleep(60)
                        continue
                    self._scroll_to_load()
                    next_data = self._extract_next_data()
                    listings = self._parse_from_json(next_data, city, state) if next_data else self._parse_from_html(city, state)
                    if not listings:
                        logger.warning(f"No listings on page {page}")
                        break
                    for li in listings:
                        if not li.listing_status:
                            li.listing_status = {"forSale": "for_sale", "forRent": "for_rent", "recentlySold": "recently_sold"}.get(status, status)
                        if status == "forRent":
                            li.monthly_rent = li.price
                            li.price = None
                    all_listings.extend(listings)
                    logger.info(f"Page {page}: {len(listings)} listings | Total: {len(all_listings)}")
                    human_delay(5, 10)
                except WebDriverException as e:
                    logger.error(f"Browser error: {e}")
                    human_delay(10, 20)
        logger.info(f"Zillow total for {city}, {state}: {len(all_listings)}")
        return all_listings


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER 2: REDFIN
# ═══════════════════════════════════════════════════════════════════════════════

class RedfinScraper:
    """
    Scrapes Redfin for property listings and market stats.
    Redfin uses numeric city IDs in their URLs.
    """

    BASE_URL = "https://www.redfin.com"

    # Redfin city IDs for our 10 target markets
    CITY_IDS = {
        "Austin": 30818,
        "Boise": 2174,
        "Raleigh": 30451,
        "Nashville": 28169,
        "Phoenix": 14240,
        "Tampa": 17516,
        "Charlotte": 3105,
        "Salt Lake City": 30789,
        "Denver": 5155,
        "Jacksonville": 8817,
        "New York": 30749,
        "Boston": 1826,
        "San Francisco": 17151,
        "Los Angeles": 11203,
        "Seattle": 16163,
        "Chicago": 17426,
        "Buffalo": 2512,
        "Hartford": 7283,
        "Durham": 5765,
        "St. Louis": 28862,
    }

    def __init__(self, driver):
        self.driver = driver

    def _get_city_id(self, city: str) -> int:
        return self.CITY_IDS.get(city, 0)

    def _build_search_url(self, city: str, state: str, status: str = "forSale") -> str:
        city_id = self._get_city_id(city)
        city_slug = city.replace(" ", "-")
        base = f"{self.BASE_URL}/city/{city_id}/{state.upper()}/{city_slug}"
        if status == "recentlySold":
            return f"{base}/filter/include=sold-3mo"
        elif status == "forRent":
            return f"{base}/apartments-for-rent"
        else:
            return base

    def _extract_redfin_data(self) -> list:
        return self._parse_redfin_html()

    def _parse_redfin_json(self, data: dict) -> list:
        listings = []
        def find_homes(obj, depth=0):
            if depth > 10: return
            if isinstance(obj, dict):
                if "price" in obj and ("address" in obj or "streetAddress" in obj):
                    listings.append(obj)
                    return
                if "homeData" in obj:
                    listings.append(obj["homeData"])
                    return
                for v in obj.values():
                    find_homes(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    find_homes(item, depth + 1)
        find_homes(data)
        return listings

    def _parse_redfin_html(self) -> list:
        listings = []
        soup = BeautifulSoup(self.driver.page_source, "lxml")

        # Find all property cards
        cards = soup.select("div.bp-Homecard__Content")
        logger.info(f"Found {len(cards)} Redfin cards")

        for card in cards:
            record = {}

            # Price
            price_el = card.select_one("div.bp-Homecard__Price")
            if price_el:
                record["price"] = parse_price(price_el.get_text())

            # Address
            addr_el = card.select_one("a.bp-Homecard__Address")
            if addr_el:
                record["address"] = addr_el.get_text(strip=True)
                href = addr_el.get("href", "")
                if not href.startswith("http"):
                    href = self.BASE_URL + href
                record["url"] = href

            # Stats (beds, baths, sqft)
            stats_el = card.select_one("div.bp-Homecard__Stats")
            if stats_el:
                stat_text = stats_el.get_text(" ")
                beds_m = re.search(r'(\d+)\s*(?:bed|bd|br)', stat_text, re.IGNORECASE)
                baths_m = re.search(r'([\d.]+)\s*(?:bath|ba)', stat_text, re.IGNORECASE)
                sqft_m = re.search(r'([\d,]+)\s*(?:sq\s*ft|sqft)', stat_text, re.IGNORECASE)
                if beds_m: record["beds"] = int(beds_m.group(1))
                if baths_m: record["baths"] = parse_float(baths_m.group(1))
                if sqft_m: record["sqft"] = parse_int(sqft_m.group(1))

            if record.get("price"):
                listings.append(record)

        logger.info(f"Extracted {len(listings)} listings from Redfin HTML")
        return listings

    def scrape_market_stats(self, city: str, state: str) -> dict:
        city_id = self._get_city_id(city)
        city_slug = city.replace(" ", "-")
        url = f"{self.BASE_URL}/city/{city_id}/{state.upper()}/{city_slug}/housing-market"
        stats = {"city": city, "state": state, "source": "redfin", "url": url}
        try:
            self.driver.get(url)
            human_delay(3, 6)
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            stat_blocks = soup.select(".MarketStat, [class*='KeyStat'], [class*='market-stat']")
            for block in stat_blocks:
                label_el = block.select_one(".label, [class*='Label'], dt")
                value_el = block.select_one(".value, [class*='Value'], dd")
                if label_el and value_el:
                    label = label_el.get_text(strip=True).lower()
                    value = value_el.get_text(strip=True)
                    if "median sale price" in label: stats["median_sale_price"] = parse_price(value)
                    elif "homes sold" in label: stats["homes_sold"] = parse_int(value)
                    elif "days on market" in label or "median days" in label: stats["median_dom"] = parse_int(value)
                    elif "sale-to-list" in label: stats["sale_to_list_ratio"] = parse_float(value)
            change_els = soup.select("[class*='change'], [class*='Change']")
            for el in change_els:
                text = el.get_text(strip=True)
                if "%" in text:
                    val = parse_float(text.replace("+", "").replace("−", "-").replace("–", "-"))
                    if "down" in text.lower() or "−" in text or "–" in text:
                        val = -abs(val) if val else None
                    if val and "median_yoy_change" not in stats:
                        stats["median_yoy_change"] = val
            logger.info(f"Redfin market stats for {city}: {stats}")
        except Exception as e:
            logger.error(f"Error scraping Redfin stats: {e}")
            stats["error"] = str(e)
        return stats

    def scrape_market(self, city: str, state: str, max_pages: int = 2) -> list:
        all_records = []
        for status in ["forSale", "recentlySold"]:
            url = self._build_search_url(city, state, status)
            logger.info(f"Scraping Redfin: {city}, {state} | {status} | {url}")
            try:
                self.driver.get(url)
                human_delay(4, 8)
                for _ in range(3):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    human_delay(1.5, 3)
                raw_listings = self._extract_redfin_data()
                for raw in raw_listings:
                    record = PropertyListing(
                        source="redfin", scrape_timestamp=datetime.now().isoformat(),
                        city=city, state=state,
                        listing_status="for_sale" if status == "forSale" else "recently_sold",
                    )
                    if isinstance(raw, dict):
                        record.price = raw.get("price") or parse_price(str(raw.get("price", "")))
                        record.full_address = raw.get("address") or raw.get("streetAddress", "")
                        record.bedrooms = raw.get("beds") or raw.get("bedrooms")
                        record.bathrooms = raw.get("baths") or raw.get("bathrooms")
                        record.sqft = raw.get("sqft") or raw.get("sqFt")
                        record.listing_url = raw.get("url", "")
                        if record.price and record.sqft and record.sqft > 0:
                            record.price_per_sqft = round(record.price / record.sqft, 2)
                        if record.full_address:
                            zip_m = re.search(r'(\d{5})', record.full_address)
                            if zip_m: record.zip_code = zip_m.group(1)
                    if record.price and record.price > 10000:
                        all_records.append(record)
                human_delay(5, 10)
            except Exception as e:
                logger.error(f"Error scraping Redfin {status}: {e}")
        logger.info(f"Redfin total for {city}, {state}: {len(all_records)} listings")
        return all_records


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER 3: REALTOR.COM
# ═══════════════════════════════════════════════════════════════════════════════

class RealtorScraper:
    BASE_URL = "https://www.realtor.com"

    def __init__(self, driver):
        self.driver = driver

    def _build_search_url(self, city: str, state: str, status: str = "buy") -> str:
        slug = f"{city.replace(' ', '-')}_{state.upper()}"
        if status == "rent":
            return f"{self.BASE_URL}/apartments/{slug}"
        elif status == "sold":
            return f"{self.BASE_URL}/realestateandhomes-search/{slug}/show-recently-sold"
        else:
            return f"{self.BASE_URL}/realestateandhomes-search/{slug}"

    def scrape_neighborhood_data(self, city: str, state: str, zip_code: str) -> dict:
        url = f"{self.BASE_URL}/neighborhoods/{city.replace(' ', '-')}_{state.upper()}"
        neighborhood = {"city": city, "state": state, "zip_code": zip_code, "source": "realtor"}
        try:
            self.driver.get(url)
            human_delay(3, 6)
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            # School ratings
            school_els = soup.select("[class*='school'], [class*='School']")
            ratings = []
            for el in school_els:
                rating_el = el.find(string=re.compile(r'\d+/10'))
                if rating_el:
                    match = re.search(r'(\d+)/10', rating_el)
                    if match: ratings.append(int(match.group(1)))
            if ratings:
                neighborhood["avg_school_rating"] = round(sum(ratings) / len(ratings), 1)
            logger.info(f"Neighborhood data for {city}, {zip_code}: {neighborhood}")
        except Exception as e:
            logger.warning(f"Error scraping neighborhood: {e}")
            neighborhood["error"] = str(e)
        return neighborhood

    def scrape_market(self, city: str, state: str, max_pages: int = 2) -> list:
        all_records = []
        for status in ["buy", "sold"]:
            url = self._build_search_url(city, state, status)
            logger.info(f"Scraping Realtor.com: {city}, {state} | {status} | {url}")
            try:
                self.driver.get(url)
                human_delay(4, 8)
                for _ in range(3):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    human_delay(1.5, 3)
                soup = BeautifulSoup(self.driver.page_source, "lxml")
                selectors = [
                    "div[data-testid='property-card']",
                    "li[data-testid='result-card']",
                    "div[class*='PropertyCard']",
                ]
                cards = []
                for sel in selectors:
                    cards = soup.select(sel)
                    if cards: break
                for card in cards:
                    record = PropertyListing(
                        source="realtor", scrape_timestamp=datetime.now().isoformat(),
                        city=city, state=state,
                        listing_status="for_sale" if status == "buy" else "recently_sold",
                    )
                    price_el = card.select_one("[data-testid='card-price'], span[class*='Price']")
                    if price_el: record.price = parse_price(price_el.get_text())
                    addr_el = card.select_one("[data-testid='card-address'], div[class*='address']")
                    if addr_el:
                        record.full_address = addr_el.get_text(strip=True)
                        zip_m = re.search(r'(\d{5})', record.full_address)
                        if zip_m: record.zip_code = zip_m.group(1)
                    text = card.get_text(" ")
                    beds_m = re.search(r'(\d+)\s*(?:bed|bd)', text, re.IGNORECASE)
                    baths_m = re.search(r'([\d.]+)\s*(?:bath|ba)', text, re.IGNORECASE)
                    sqft_m = re.search(r'([\d,]+)\s*(?:sqft|sq ft)', text, re.IGNORECASE)
                    if beds_m: record.bedrooms = int(beds_m.group(1))
                    if baths_m: record.bathrooms = parse_float(baths_m.group(1))
                    if sqft_m: record.sqft = parse_int(sqft_m.group(1))
                    if record.price and record.sqft and record.sqft > 0:
                        record.price_per_sqft = round(record.price / record.sqft, 2)
                    if record.price and record.price > 10000:
                        all_records.append(record)
                human_delay(5, 10)
            except Exception as e:
                logger.error(f"Error scraping Realtor.com {status}: {e}")
        logger.info(f"Realtor.com total for {city}, {state}: {len(all_records)} listings")
        return all_records


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER 4: CENSUS BUREAU API
# ═══════════════════════════════════════════════════════════════════════════════

class CensusCollector:
    ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
    VARIABLES = "B19013_001E,B01003_001E,B25077_001E,B25064_001E"

    def get_income_by_zip(self, zip_codes: list) -> list:
        records = []
        for zip_code in zip_codes:
            url = f"{self.ACS_URL}?get=NAME,{self.VARIABLES}&for=zip%20code%20tabulation%20area:{zip_code}"
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                if len(data) > 1:
                    v = data[1]
                    records.append({
                        "zip_code": zip_code,
                        "area_name": v[0],
                        "median_household_income": self._safe_int(v[1]),
                        "population": self._safe_int(v[2]),
                        "census_median_home_value": self._safe_int(v[3]),
                        "census_median_rent": self._safe_int(v[4]),
                    })
                human_delay(0.3, 0.8)
            except Exception as e:
                logger.warning(f"Census error for {zip_code}: {e}")
        logger.info(f"Census data: {len(records)}/{len(zip_codes)} ZIP codes")
        return records

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        try:
            val = int(value)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class DataCollectionPipeline:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.all_properties = []
        self.all_market_stats = []
        self.all_income_data = []
        self.all_neighborhood_data = []

    def _init_browser(self):
        if self.driver:
            try: self.driver.quit()
            except: pass
        self.driver = BrowserFactory.create(headless=self.headless)

    def run(self, markets=None, max_pages=3, scrape_details=False, sources=None):
        if markets is None: markets = TARGET_MARKETS
        if sources is None: sources = ["zillow", "redfin", "realtor", "census"]

        logger.info(f"\n{'═'*60}")
        logger.info(f"DATA COLLECTION PIPELINE")
        logger.info(f"Markets: {len(markets)} | Pages: {max_pages} | Sources: {sources}")
        logger.info(f"{'═'*60}")

        start_time = datetime.now()

        if any(s in sources for s in ["zillow", "redfin", "realtor"]):
            self._init_browser()

        census = CensusCollector()

        for i, market in enumerate(markets):
            city, state = market["city"], market["state"]
            zip_codes = market.get("zip_codes", [])

            logger.info(f"\n{'━'*60}")
            logger.info(f"[{i+1}/{len(markets)}] Processing: {city}, {state}")
            logger.info(f"{'━'*60}")

            if "zillow" in sources:
                try:
                    zillow = ZillowScraper(self.driver)
                    self.all_properties.extend(zillow.scrape_market(city, state, max_pages, scrape_details=scrape_details))
                except Exception as e:
                    logger.error(f"Zillow failed for {city}: {e}")
                if (i + 1) % 3 == 0:
                    self._init_browser()
                    human_delay(5, 10)

            if "redfin" in sources:
                try:
                    redfin = RedfinScraper(self.driver)
                    self.all_properties.extend(redfin.scrape_market(city, state, max_pages))
                    self.all_market_stats.append(redfin.scrape_market_stats(city, state))
                except Exception as e:
                    logger.error(f"Redfin failed for {city}: {e}")

            if "realtor" in sources:
                try:
                    realtor = RealtorScraper(self.driver)
                    self.all_properties.extend(realtor.scrape_market(city, state, max_pages=1))
                    for zc in zip_codes[:3]:
                        self.all_neighborhood_data.append(realtor.scrape_neighborhood_data(city, state, zc))
                        human_delay(3, 6)
                except Exception as e:
                    logger.error(f"Realtor.com failed for {city}: {e}")

            if "census" in sources and zip_codes:
                income_data = census.get_income_by_zip(zip_codes)
                for r in income_data:
                    r["city"] = city
                    r["state"] = state
                self.all_income_data.extend(income_data)

            if i < len(markets) - 1:
                pause = random.uniform(15, 30)
                logger.info(f"Pausing {pause:.0f}s before next market...")
                time.sleep(pause)

        self._save_all()
        if self.driver: self.driver.quit()

        elapsed = (datetime.now() - start_time).total_seconds() / 60
        self._print_report(elapsed)

    def _save_all(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.all_properties:
            save_to_csv(self.all_properties, f"properties_raw_{ts}.csv")
            save_to_csv(self.all_properties, "properties_raw_latest.csv")
        if self.all_market_stats:
            fp = RAW_DIR / f"market_stats_{ts}.csv"
            with open(fp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.all_market_stats[0].keys())
                w.writeheader()
                w.writerows(self.all_market_stats)
            logger.info(f"Saved {len(self.all_market_stats)} market stats")
        if self.all_income_data:
            fp = RAW_DIR / f"census_income_{ts}.csv"
            with open(fp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.all_income_data[0].keys())
                w.writeheader()
                w.writerows(self.all_income_data)
            logger.info(f"Saved {len(self.all_income_data)} income records")
        if self.all_neighborhood_data:
            fp = RAW_DIR / f"neighborhoods_{ts}.csv"
            keys = set()
            for d in self.all_neighborhood_data: keys.update(d.keys())
            with open(fp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=sorted(keys))
                w.writeheader()
                w.writerows(self.all_neighborhood_data)
            logger.info(f"Saved {len(self.all_neighborhood_data)} neighborhood records")

    def _print_report(self, elapsed_minutes: float):
        logger.info(f"\n{'═'*60}")
        logger.info(f"COLLECTION REPORT")
        logger.info(f"{'═'*60}")
        logger.info(f"Duration:         {elapsed_minutes:.1f} minutes")
        logger.info(f"Total Properties: {len(self.all_properties)}")
        if self.all_properties:
            sources = {}
            for p in self.all_properties: sources[p.source] = sources.get(p.source, 0) + 1
            for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
                logger.info(f"  {src}: {cnt}")
            cities = {}
            for p in self.all_properties:
                k = f"{p.city}, {p.state}"
                cities[k] = cities.get(k, 0) + 1
            for c, cnt in sorted(cities.items(), key=lambda x: -x[1]):
                logger.info(f"  {c}: {cnt}")
        logger.info(f"Market Stats:     {len(self.all_market_stats)}")
        logger.info(f"Income Records:   {len(self.all_income_data)}")
        logger.info(f"Neighborhoods:    {len(self.all_neighborhood_data)}")
        logger.info(f"\nRaw data saved to: {RAW_DIR}/")
        logger.info(f"✅ Done! Next: python scripts/02_data_cleaning.py")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape U.S. real estate data")
    parser.add_argument("--market", action="append", default=[], help="'City, ST'")
    parser.add_argument("--all", action="store_true", help="All 10 markets")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--sources", nargs="+", default=["zillow", "redfin", "realtor", "census"],
                        choices=["zillow", "redfin", "realtor", "census"])
    args = parser.parse_args()

    if args.all:
        markets = TARGET_MARKETS
    elif args.market:
        markets = []
        for m in args.market:
            parts = m.split(",")
            city = parts[0].strip()
            state = parts[1].strip() if len(parts) > 1 else ""
            match = next((t for t in TARGET_MARKETS if t["city"].lower() == city.lower()),
                         {"city": city, "state": state, "zip_codes": []})
            markets.append(match)
    else:
        print("Specify --market 'City, ST' or --all")
        exit(1)

    pipeline = DataCollectionPipeline(headless=not args.no_headless)
    pipeline.run(markets=markets, max_pages=args.pages, scrape_details=args.details, sources=args.sources)