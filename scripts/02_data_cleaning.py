"""
02_data_cleaning.py
===================
Cleans and merges ALL data sources into one enriched dataset.

Data Sources:
  FROM data/raw/ (your scraper output):
    - properties_raw_latest.csv      → 706 property listings
    - census_income_*.csv            → 50 ZIP-level income/rent records
    - market_stats_*.csv             → 10 scraped market stats (limited)
    - neighborhoods_*.csv            → 30 neighborhood records

  FROM data/supplementary/ (01b script output):
    - redfin_market_stats_latest.csv → 10 cities: DOM, prices, inventory
    - redfin_historical_latest.csv   → 10 cities: YoY price growth
    - crime_data_latest.csv          → 10 cities: crime rates, safety score
    - school_ratings_latest.csv      → 10 cities: school ratings

Output (to data/cleaned/):
    - properties_cleaned.csv         → Fully merged, one row per property

Usage:
  python scripts/02_data_cleaning.py
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
SUPP_DIR = Path("data/supplementary")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_latest(pattern: str, directory: Path) -> Path:
    matches = sorted(
        glob.glob(str(directory / pattern)),
        key=os.path.getmtime, reverse=True,
    )
    return Path(matches[0]) if matches else None


def load_csv(pattern: str, directory: Path, label: str) -> pd.DataFrame:
    path = find_latest(pattern, directory)
    if path and path.exists():
        df = pd.read_csv(path, dtype=str)
        logger.info(f"  ✓ {label}: {len(df)} rows from {path.name}")
        return df
    logger.warning(f"  ✗ {label}: not found ({pattern})")
    return pd.DataFrame()


# Valid ZIP prefixes per city (to fix scraping artifacts like '10616' for Austin)
VALID_ZIP_PREFIXES = {
    "Austin": ["787"], "Boise": ["837"], "Raleigh": ["276"],
    "Nashville": ["372"], "Phoenix": ["850"], "Tampa": ["336", "335"],
    "Charlotte": ["282"], "Salt Lake City": ["841"],
    "Denver": ["802"], "Jacksonville": ["322"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data() -> dict:
    logger.info("Loading all data sources...")
    data = {}

    # ── From data/raw/ ──
    logger.info("\nFrom data/raw/:")
    data["properties"] = load_csv("properties_raw_latest.csv", RAW_DIR, "Properties")
    if data["properties"].empty:
        data["properties"] = load_csv("properties_raw_*.csv", RAW_DIR, "Properties (fallback)")
    data["census"] = load_csv("census_income_*.csv", RAW_DIR, "Census income")
    data["neighborhoods"] = load_csv("neighborhoods_*.csv", RAW_DIR, "Neighborhoods")

    # ── From data/supplementary/ ──
    logger.info("\nFrom data/supplementary/:")
    data["redfin_stats"] = load_csv("redfin_market_stats_latest.csv", SUPP_DIR, "Redfin market stats")
    data["redfin_historical"] = load_csv("redfin_historical_latest.csv", SUPP_DIR, "Redfin historical")
    data["crime"] = load_csv("crime_data_latest.csv", SUPP_DIR, "Crime data")
    data["schools"] = load_csv("school_ratings_latest.csv", SUPP_DIR, "School ratings")
    data["walkscore"] = load_csv("walk_scores_latest.csv", SUPP_DIR, "Walk scores")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

def clean_properties(df: pd.DataFrame) -> tuple:
    logger.info(f"\nCleaning properties: {len(df)} rows")
    report = {"initial_rows": len(df)}
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Drop 100% empty columns (scraper didn't capture these)
    empty_cols = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[~vals.isin(["", "nan", "None", "NaN"])]
        if len(vals) == 0:
            empty_cols.append(col)
    if empty_cols:
        logger.info(f"Dropping {len(empty_cols)} empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # Cast numeric
    numeric_cols = [
        "price", "original_price", "monthly_rent", "bathrooms",
        "price_per_sqft", "hoa_fee", "zestimate", "rent_zestimate",
        "tax_assessed_value", "annual_tax", "latitude", "longitude",
        "bedrooms", "sqft", "lot_sqft", "year_built", "days_on_market",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
                errors="coerce",
            )

    # Standardize text
    if "city" in df.columns:
        df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
    if "source" in df.columns:
        df["source"] = df["source"].str.strip().str.lower()

    # Fix ZIP codes
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.strip().str.extract(r"(\d{5})", expand=False)
        invalid = 0
        for city, prefixes in VALID_ZIP_PREFIXES.items():
            city_mask = df["city"] == city
            if city_mask.any():
                bad = city_mask & df["zip_code"].notna() & ~df["zip_code"].str[:3].isin(prefixes)
                invalid += bad.sum()
                df.loc[bad, "zip_code"] = pd.NA
        if invalid:
            logger.info(f"Fixed {invalid} invalid ZIP codes")

    # Standardize listing_status
    if "listing_status" in df.columns:
        status_map = {
            "for sale": "for_sale", "forsale": "for_sale", "for_sale": "for_sale",
            "recently sold": "recently_sold", "recentlysold": "recently_sold",
            "recently_sold": "recently_sold", "sold": "recently_sold",
            "for rent": "for_rent", "forrent": "for_rent", "for_rent": "for_rent",
        }
        df["listing_status"] = df["listing_status"].str.strip().str.lower().map(status_map).fillna("unknown")

    # Standardize property_type
    if "property_type" in df.columns:
        type_map = {
            "single_family": "single_family", "single family": "single_family",
            "house": "single_family", "condo": "condo", "condominium": "condo",
            "townhouse": "townhouse", "townhome": "townhouse",
            "multi_family": "multi_family", "apartment": "apartment",
            "lot": "land", "land": "land", "manufactured": "manufactured",
        }
        df["property_type"] = df["property_type"].str.strip().str.lower().map(type_map).fillna("other")

    # Drop rows without price
    before = len(df)
    df = df.dropna(subset=["price"])
    report["dropped_no_price"] = before - len(df)
    logger.info(f"Dropped {before - len(df)} rows with no price")

    # Price outliers
    before = len(df)
    df = df[(df["price"] >= 20_000) & (df["price"] <= 10_000_000)]
    report["price_outliers"] = before - len(df)
    logger.info(f"Removed {before - len(df)} price outliers")

    # Null out bad values (don't drop rows)
    if "sqft" in df.columns:
        bad = (df["sqft"] < 100) | (df["sqft"] > 15_000)
        df.loc[bad & df["sqft"].notna(), "sqft"] = np.nan
    if "bedrooms" in df.columns:
        df.loc[df["bedrooms"] > 10, "bedrooms"] = np.nan
    if "bathrooms" in df.columns:
        df.loc[df["bathrooms"] > 10, "bathrooms"] = np.nan

    # Impute missing beds/baths/sqft with city median
    for col in ["bedrooms", "bathrooms"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("city")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    if "sqft" in df.columns and df["sqft"].isna().any():
        df["sqft"] = df.groupby("city")["sqft"].transform(lambda x: x.fillna(x.median()))
        df["sqft"] = df["sqft"].fillna(df["sqft"].median())

    # Deduplicate
    before_dedup = len(df)
    df = df.drop_duplicates()
    if "full_address" in df.columns and "source" in df.columns:
        df = df.drop_duplicates(subset=["full_address", "price", "source"], keep="first")
        source_rank = {"redfin": 0, "realtor": 1}
        df["_rank"] = df["source"].map(source_rank).fillna(9)
        df = df.sort_values("_rank").drop_duplicates(subset=["full_address"], keep="first")
        df = df.drop(columns=["_rank"])
    report["duplicates_removed"] = before_dedup - len(df)
    logger.info(f"Removed {before_dedup - len(df)} duplicates")

    # Recalculate price_per_sqft
    if "sqft" in df.columns:
        mask = df["price"].notna() & df["sqft"].notna() & (df["sqft"] > 0)
        df.loc[mask, "price_per_sqft"] = (df.loc[mask, "price"] / df.loc[mask, "sqft"]).round(2)

    df = df.reset_index(drop=True)
    report["final_rows"] = len(df)
    logger.info(f"Properties cleaned: {report['initial_rows']} → {report['final_rows']} rows")
    return df, report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLEAN SUPPLEMENTARY DATA
# ═══════════════════════════════════════════════════════════════════════════════

def clean_census(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    logger.info(f"Cleaning census: {len(df)} rows")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    if "city" in df.columns:
        df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
    for col in ["median_household_income", "population", "census_median_home_value", "census_median_rent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
    df = df.drop_duplicates(subset=["zip_code"], keep="first")
    return df


def clean_supplementary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Generic cleaner for city-level supplementary CSVs."""
    if df.empty:
        return df
    logger.info(f"Cleaning {label}: {len(df)} rows")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if "city" in df.columns:
        df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
    # Cast all numeric-looking columns
    for col in df.columns:
        if col not in ["city", "state", "source", "period_end", "school_density",
                        "walk_desc", "transit_desc", "bike_desc", "zip_code"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MERGE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_all(properties: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Merge all data sources onto the property-level data.

    Join strategy:
      - Census         → on zip_code (with city-level fallback)
      - Redfin stats   → on city + state
      - Redfin history  → on city + state
      - Crime           → on city + state
      - Schools         → on city + state
      - Walk scores     → on zip_code (if available)
      - Neighborhoods   → on city + zip_code (if available)
    """
    logger.info(f"\nMerging all datasets onto {len(properties)} properties...")
    merged = properties.copy()

    # ── Census (ZIP-level, with city fallback) ────────────────────────────
    census = data.get("census", pd.DataFrame())
    if not census.empty and "zip_code" in merged.columns:
        census_cols = ["zip_code", "median_household_income", "population",
                       "census_median_home_value", "census_median_rent"]
        available = [c for c in census_cols if c in census.columns]

        merged = merged.merge(census[available], on="zip_code", how="left")
        zip_match = merged["median_household_income"].notna().sum()
        logger.info(f"  Census ZIP match: {zip_match}/{len(merged)}")

        # City-level fallback for unmatched rows
        if zip_match < len(merged) and "city" in census.columns:
            city_medians = census.groupby("city").agg({
                c: "median" for c in available if c != "zip_code"
            }).reset_index()
            for col in [c for c in available if c != "zip_code"]:
                if col in city_medians.columns:
                    city_map = city_medians.set_index("city")[col].to_dict()
                    missing = merged[col].isna()
                    merged.loc[missing, col] = merged.loc[missing, "city"].map(city_map)
            final_match = merged["median_household_income"].notna().sum()
            logger.info(f"  Census after city fallback: {final_match}/{len(merged)}")

    # ── Redfin Market Stats (city-level) ──────────────────────────────────
    redfin_stats = data.get("redfin_stats", pd.DataFrame())
    if not redfin_stats.empty:
        # Drop 'source' column to avoid conflicts
        drop = [c for c in ["source", "period_end"] if c in redfin_stats.columns]
        stats_merge = redfin_stats.drop(columns=drop, errors="ignore")
        merged = merged.merge(stats_merge, on=["city", "state"], how="left")
        matched = merged["median_dom"].notna().sum() if "median_dom" in merged.columns else 0
        logger.info(f"  Redfin stats match: {matched}/{len(merged)}")

    # ── Redfin Historical / YoY Growth (city-level) ───────────────────────
    redfin_hist = data.get("redfin_historical", pd.DataFrame())
    if not redfin_hist.empty:
        drop = [c for c in ["latest_period"] if c in redfin_hist.columns]
        hist_merge = redfin_hist.drop(columns=drop, errors="ignore")
        merged = merged.merge(hist_merge, on=["city", "state"], how="left")
        matched = merged["yoy_price_change_pct"].notna().sum() if "yoy_price_change_pct" in merged.columns else 0
        logger.info(f"  YoY growth match: {matched}/{len(merged)}")

    # ── Crime Data (city-level) ───────────────────────────────────────────
    crime = data.get("crime", pd.DataFrame())
    if not crime.empty:
        drop = [c for c in ["source"] if c in crime.columns]
        crime_merge = crime.drop(columns=drop, errors="ignore")
        merged = merged.merge(crime_merge, on=["city", "state"], how="left")
        matched = merged["safety_score"].notna().sum() if "safety_score" in merged.columns else 0
        logger.info(f"  Crime data match: {matched}/{len(merged)}")

    # ── School Ratings (city-level) ───────────────────────────────────────
    schools = data.get("schools", pd.DataFrame())
    if not schools.empty:
        merged = merged.merge(schools, on=["city", "state"], how="left")
        matched = merged["avg_school_rating"].notna().sum() if "avg_school_rating" in merged.columns else 0
        logger.info(f"  School ratings match: {matched}/{len(merged)}")

    # ── Walk Scores (ZIP-level, optional) ─────────────────────────────────
    walkscore = data.get("walkscore", pd.DataFrame())
    if not walkscore.empty and "zip_code" in walkscore.columns and "zip_code" in merged.columns:
        ws_cols = ["zip_code", "walk_score", "transit_score", "bike_score"]
        available = [c for c in ws_cols if c in walkscore.columns]
        merged = merged.merge(walkscore[available], on="zip_code", how="left")
        matched = merged["walk_score"].notna().sum() if "walk_score" in merged.columns else 0
        logger.info(f"  Walk score match: {matched}/{len(merged)}")

    # ── Neighborhoods (city + ZIP, optional) ──────────────────────────────
    nbr = data.get("neighborhoods", pd.DataFrame())
    if not nbr.empty:
        nbr = nbr.copy()
        nbr.columns = nbr.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        if "city" in nbr.columns:
            nbr["city"] = nbr["city"].str.strip().str.title()
        if "zip_code" in nbr.columns:
            nbr["zip_code"] = nbr["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
        # Only merge if it has data columns beyond keys
        nbr_data_cols = [c for c in nbr.columns if c not in ["city", "state", "zip_code", "source", "error"]]
        if nbr_data_cols and "zip_code" in nbr.columns and "zip_code" in merged.columns:
            # Avoid duplicating avg_school_rating if already merged from schools
            for col in nbr_data_cols:
                if col in merged.columns:
                    nbr = nbr.drop(columns=[col])
            nbr_data_cols = [c for c in nbr.columns if c not in ["city", "state", "zip_code", "source", "error"]]
            if nbr_data_cols:
                merge_cols = ["city", "zip_code"] + nbr_data_cols
                available = [c for c in merge_cols if c in nbr.columns]
                merged = merged.merge(nbr[available], on=["city", "zip_code"], how="left")
                logger.info(f"  Neighborhood merge: done")

    logger.info(f"\nFinal merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 5. QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame):
    logger.info(f"\n{'─'*60}")
    logger.info(f"DATA QUALITY REPORT")
    logger.info(f"{'─'*60}")
    logger.info(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")

    # Columns with data
    logger.info(f"\nColumns with data:")
    for col in df.columns:
        filled = df[col].notna().sum()
        pct = round(filled / len(df) * 100, 1)
        if pct > 0:
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            logger.info(f"  {col:<35} {filled:>5}/{len(df)}  ({pct:>5.1f}%) {bar}")

    # Numeric stats
    numeric = df.select_dtypes(include=[np.number])
    has_data = numeric.columns[numeric.notna().any()]
    if len(has_data) > 0:
        logger.info(f"\nKey numeric stats:")
        for col in has_data:
            vals = df[col].dropna()
            if len(vals) > 0:
                logger.info(f"  {col:<35} min={vals.min():>12,.1f}  median={vals.median():>12,.1f}  max={vals.max():>12,.1f}")

    # Records per city
    if "city" in df.columns:
        logger.info(f"\nRecords per city:")
        for city, count in df["city"].value_counts().items():
            logger.info(f"  {city}: {count}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("DATA CLEANING + MERGE PIPELINE")
    logger.info("=" * 60)

    # ── Load everything ───────────────────────────────────────────────────
    data = load_all_data()

    if data["properties"].empty:
        logger.error("No property data found. Run 01_data_collection.py first.")
        return

    # ── Clean properties ──────────────────────────────────────────────────
    properties, report = clean_properties(data["properties"])

    # ── Clean supplementary data ──────────────────────────────────────────
    logger.info("\nCleaning supplementary data...")
    clean_data = {
        "census": clean_census(data.get("census", pd.DataFrame())),
        "redfin_stats": clean_supplementary(data.get("redfin_stats", pd.DataFrame()), "Redfin stats"),
        "redfin_historical": clean_supplementary(data.get("redfin_historical", pd.DataFrame()), "Redfin historical"),
        "crime": clean_supplementary(data.get("crime", pd.DataFrame()), "Crime data"),
        "schools": clean_supplementary(data.get("schools", pd.DataFrame()), "School ratings"),
        "walkscore": clean_supplementary(data.get("walkscore", pd.DataFrame()), "Walk scores"),
        "neighborhoods": data.get("neighborhoods", pd.DataFrame()),
    }

    # ── Merge ─────────────────────────────────────────────────────────────
    merged = merge_all(properties, clean_data)

    # ── Save ──────────────────────────────────────────────────────────────
    merged.to_csv(CLEAN_DIR / "properties_cleaned.csv", index=False)
    logger.info(f"\nSaved: properties_cleaned.csv ({len(merged)} rows, {len(merged.columns)} cols)")

    # Also save cleaned supplementary files for reference
    for name, df in clean_data.items():
        if not df.empty:
            df.to_csv(CLEAN_DIR / f"{name}_cleaned.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────
    print_report(merged)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Properties:    {report['initial_rows']} → {report['final_rows']} rows")
    logger.info(f"  No price:    {report.get('dropped_no_price', 0)}")
    logger.info(f"  Outliers:    {report.get('price_outliers', 0)}")
    logger.info(f"  Duplicates:  {report.get('duplicates_removed', 0)}")
    logger.info(f"Final dataset: {len(merged)} rows × {len(merged.columns)} columns")
    logger.info(f"\n✅ Done! Next: python scripts/03_feature_engineering.py")


if __name__ == "__main__":
    main()
    """
02_data_cleaning.py
===================
Cleans and merges ALL data sources into one enriched dataset.

Data Sources:
  FROM data/raw/ (your scraper output):
    - properties_raw_latest.csv      → 706 property listings
    - census_income_*.csv            → 50 ZIP-level income/rent records
    - market_stats_*.csv             → 10 scraped market stats (limited)
    - neighborhoods_*.csv            → 30 neighborhood records

  FROM data/supplementary/ (01b script output):
    - redfin_market_stats_latest.csv → 10 cities: DOM, prices, inventory
    - redfin_historical_latest.csv   → 10 cities: YoY price growth
    - crime_data_latest.csv          → 10 cities: crime rates, safety score
    - school_ratings_latest.csv      → 10 cities: school ratings

Output (to data/cleaned/):
    - properties_cleaned.csv         → Fully merged, one row per property

Usage:
  python scripts/02_data_cleaning.py
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
SUPP_DIR = Path("data/supplementary")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_latest(pattern: str, directory: Path) -> Path:
    matches = sorted(
        glob.glob(str(directory / pattern)),
        key=os.path.getmtime, reverse=True,
    )
    return Path(matches[0]) if matches else None


def load_csv(pattern: str, directory: Path, label: str) -> pd.DataFrame:
    path = find_latest(pattern, directory)
    if path and path.exists():
        df = pd.read_csv(path, dtype=str)
        logger.info(f"  ✓ {label}: {len(df)} rows from {path.name}")
        return df
    logger.warning(f"  ✗ {label}: not found ({pattern})")
    return pd.DataFrame()


# Valid ZIP prefixes per city (to fix scraping artifacts like '10616' for Austin)
VALID_ZIP_PREFIXES = {
    "Austin": ["787"], "Boise": ["837"], "Raleigh": ["276"],
    "Nashville": ["372"], "Phoenix": ["850"], "Tampa": ["336", "335"],
    "Charlotte": ["282"], "Salt Lake City": ["841"],
    "Denver": ["802"], "Jacksonville": ["322"],    "New York": ["100", "101", "102", "103", "104", "110", "111", "112", "113", "114"],
    "Boston": ["021"],
    "San Francisco": ["941"],
    "Los Angeles": ["900", "901", "902"],
    "Seattle": ["981"],
    "Chicago": ["606"],
    "Buffalo": ["142"],
    "Hartford": ["061"],
    "Durham": ["277"],
    "St. Louis": ["631"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data() -> dict:
    logger.info("Loading all data sources...")
    data = {}

    # ── From data/raw/ ──
    logger.info("\nFrom data/raw/:")
    data["properties"] = load_csv("properties_raw_latest.csv", RAW_DIR, "Properties")
    if data["properties"].empty:
        data["properties"] = load_csv("properties_raw_*.csv", RAW_DIR, "Properties (fallback)")
    data["census"] = load_csv("census_income_*.csv", RAW_DIR, "Census income")
    data["neighborhoods"] = load_csv("neighborhoods_*.csv", RAW_DIR, "Neighborhoods")

    # ── From data/supplementary/ ──
    logger.info("\nFrom data/supplementary/:")
    data["redfin_stats"] = load_csv("redfin_market_stats_latest.csv", SUPP_DIR, "Redfin market stats")
    data["redfin_historical"] = load_csv("redfin_historical_latest.csv", SUPP_DIR, "Redfin historical")
    data["crime"] = load_csv("crime_data_latest.csv", SUPP_DIR, "Crime data")
    data["schools"] = load_csv("school_ratings_latest.csv", SUPP_DIR, "School ratings")
    data["walkscore"] = load_csv("walk_scores_latest.csv", SUPP_DIR, "Walk scores")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

def clean_properties(df: pd.DataFrame) -> tuple:
    logger.info(f"\nCleaning properties: {len(df)} rows")
    report = {"initial_rows": len(df)}
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Drop 100% empty columns (scraper didn't capture these)
    empty_cols = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[~vals.isin(["", "nan", "None", "NaN"])]
        if len(vals) == 0:
            empty_cols.append(col)
    if empty_cols:
        logger.info(f"Dropping {len(empty_cols)} empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # Cast numeric
    numeric_cols = [
        "price", "original_price", "monthly_rent", "bathrooms",
        "price_per_sqft", "hoa_fee", "zestimate", "rent_zestimate",
        "tax_assessed_value", "annual_tax", "latitude", "longitude",
        "bedrooms", "sqft", "lot_sqft", "year_built", "days_on_market",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
                errors="coerce",
            )

    # Standardize text
    if "city" in df.columns:
        df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
    if "source" in df.columns:
        df["source"] = df["source"].str.strip().str.lower()

    # Fix ZIP codes
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.strip().str.extract(r"(\d{5})", expand=False)
        invalid = 0
        for city, prefixes in VALID_ZIP_PREFIXES.items():
            city_mask = df["city"] == city
            if city_mask.any():
                bad = city_mask & df["zip_code"].notna() & ~df["zip_code"].str[:3].isin(prefixes)
                invalid += bad.sum()
                df.loc[bad, "zip_code"] = pd.NA
        if invalid:
            logger.info(f"Fixed {invalid} invalid ZIP codes")

    # Standardize listing_status
    if "listing_status" in df.columns:
        status_map = {
            "for sale": "for_sale", "forsale": "for_sale", "for_sale": "for_sale",
            "recently sold": "recently_sold", "recentlysold": "recently_sold",
            "recently_sold": "recently_sold", "sold": "recently_sold",
            "for rent": "for_rent", "forrent": "for_rent", "for_rent": "for_rent",
        }
        df["listing_status"] = df["listing_status"].str.strip().str.lower().map(status_map).fillna("unknown")

    # Standardize property_type
    if "property_type" in df.columns:
        type_map = {
            "single_family": "single_family", "single family": "single_family",
            "house": "single_family", "condo": "condo", "condominium": "condo",
            "townhouse": "townhouse", "townhome": "townhouse",
            "multi_family": "multi_family", "apartment": "apartment",
            "lot": "land", "land": "land", "manufactured": "manufactured",
        }
        df["property_type"] = df["property_type"].str.strip().str.lower().map(type_map).fillna("other")

    # Drop rows without price
    before = len(df)
    df = df.dropna(subset=["price"])
    report["dropped_no_price"] = before - len(df)
    logger.info(f"Dropped {before - len(df)} rows with no price")

    # Price outliers
    before = len(df)
    df = df[(df["price"] >= 20_000) & (df["price"] <= 10_000_000)]
    report["price_outliers"] = before - len(df)
    logger.info(f"Removed {before - len(df)} price outliers")

    # Null out bad values (don't drop rows)
    if "sqft" in df.columns:
        bad = (df["sqft"] < 100) | (df["sqft"] > 15_000)
        df.loc[bad & df["sqft"].notna(), "sqft"] = np.nan
    if "bedrooms" in df.columns:
        df.loc[df["bedrooms"] > 10, "bedrooms"] = np.nan
    if "bathrooms" in df.columns:
        df.loc[df["bathrooms"] > 10, "bathrooms"] = np.nan

    # Impute missing beds/baths/sqft with city median
    for col in ["bedrooms", "bathrooms"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("city")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    if "sqft" in df.columns and df["sqft"].isna().any():
        df["sqft"] = df.groupby("city")["sqft"].transform(lambda x: x.fillna(x.median()))
        df["sqft"] = df["sqft"].fillna(df["sqft"].median())

    # Deduplicate
    before_dedup = len(df)
    df = df.drop_duplicates()
    if "full_address" in df.columns and "source" in df.columns:
        df = df.drop_duplicates(subset=["full_address", "price", "source"], keep="first")
        source_rank = {"redfin": 0, "realtor": 1}
        df["_rank"] = df["source"].map(source_rank).fillna(9)
        df = df.sort_values("_rank").drop_duplicates(subset=["full_address"], keep="first")
        df = df.drop(columns=["_rank"])
    report["duplicates_removed"] = before_dedup - len(df)
    logger.info(f"Removed {before_dedup - len(df)} duplicates")

    # Recalculate price_per_sqft
    if "sqft" in df.columns:
        mask = df["price"].notna() & df["sqft"].notna() & (df["sqft"] > 0)
        df.loc[mask, "price_per_sqft"] = (df.loc[mask, "price"] / df.loc[mask, "sqft"]).round(2)

    df = df.reset_index(drop=True)
    report["final_rows"] = len(df)
    logger.info(f"Properties cleaned: {report['initial_rows']} → {report['final_rows']} rows")
    return df, report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLEAN SUPPLEMENTARY DATA
# ═══════════════════════════════════════════════════════════════════════════════

def clean_census(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    logger.info(f"Cleaning census: {len(df)} rows")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    if "city" in df.columns:
        df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
    for col in ["median_household_income", "population", "census_median_home_value", "census_median_rent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
    df = df.drop_duplicates(subset=["zip_code"], keep="first")
    return df


def clean_supplementary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Generic cleaner for city-level supplementary CSVs."""
    if df.empty:
        return df
    logger.info(f"Cleaning {label}: {len(df)} rows")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if "city" in df.columns:
        df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
    # Cast all numeric-looking columns
    for col in df.columns:
        if col not in ["city", "state", "source", "period_end", "school_density",
                        "walk_desc", "transit_desc", "bike_desc", "zip_code"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MERGE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_all(properties: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Merge all data sources onto the property-level data.

    Join strategy:
      - Census         → on zip_code (with city-level fallback)
      - Redfin stats   → on city + state
      - Redfin history  → on city + state
      - Crime           → on city + state
      - Schools         → on city + state
      - Walk scores     → on zip_code (if available)
      - Neighborhoods   → on city + zip_code (if available)
    """
    logger.info(f"\nMerging all datasets onto {len(properties)} properties...")
    merged = properties.copy()

    # ── Census (ZIP-level, with city fallback) ────────────────────────────
    census = data.get("census", pd.DataFrame())
    if not census.empty and "zip_code" in merged.columns:
        census_cols = ["zip_code", "median_household_income", "population",
                       "census_median_home_value", "census_median_rent"]
        available = [c for c in census_cols if c in census.columns]

        merged = merged.merge(census[available], on="zip_code", how="left")
        zip_match = merged["median_household_income"].notna().sum()
        logger.info(f"  Census ZIP match: {zip_match}/{len(merged)}")

        # City-level fallback for unmatched rows
        if zip_match < len(merged) and "city" in census.columns:
            city_medians = census.groupby("city").agg({
                c: "median" for c in available if c != "zip_code"
            }).reset_index()
            for col in [c for c in available if c != "zip_code"]:
                if col in city_medians.columns:
                    city_map = city_medians.set_index("city")[col].to_dict()
                    missing = merged[col].isna()
                    merged.loc[missing, col] = merged.loc[missing, "city"].map(city_map)
            final_match = merged["median_household_income"].notna().sum()
            logger.info(f"  Census after city fallback: {final_match}/{len(merged)}")

    # ── Redfin Market Stats (city-level) ──────────────────────────────────
    redfin_stats = data.get("redfin_stats", pd.DataFrame())
    if not redfin_stats.empty:
        # Drop 'source' column to avoid conflicts
        drop = [c for c in ["source", "period_end"] if c in redfin_stats.columns]
        stats_merge = redfin_stats.drop(columns=drop, errors="ignore")
        merged = merged.merge(stats_merge, on=["city", "state"], how="left")
        matched = merged["median_dom"].notna().sum() if "median_dom" in merged.columns else 0
        logger.info(f"  Redfin stats match: {matched}/{len(merged)}")

    # ── Redfin Historical / YoY Growth (city-level) ───────────────────────
    redfin_hist = data.get("redfin_historical", pd.DataFrame())
    if not redfin_hist.empty:
        drop = [c for c in ["latest_period"] if c in redfin_hist.columns]
        hist_merge = redfin_hist.drop(columns=drop, errors="ignore")
        merged = merged.merge(hist_merge, on=["city", "state"], how="left")
        matched = merged["yoy_price_change_pct"].notna().sum() if "yoy_price_change_pct" in merged.columns else 0
        logger.info(f"  YoY growth match: {matched}/{len(merged)}")

    # ── Crime Data (city-level) ───────────────────────────────────────────
    crime = data.get("crime", pd.DataFrame())
    if not crime.empty:
        drop = [c for c in ["source"] if c in crime.columns]
        crime_merge = crime.drop(columns=drop, errors="ignore")
        merged = merged.merge(crime_merge, on=["city", "state"], how="left")
        matched = merged["safety_score"].notna().sum() if "safety_score" in merged.columns else 0
        logger.info(f"  Crime data match: {matched}/{len(merged)}")

    # ── School Ratings (city-level) ───────────────────────────────────────
    schools = data.get("schools", pd.DataFrame())
    if not schools.empty:
        merged = merged.merge(schools, on=["city", "state"], how="left")
        matched = merged["avg_school_rating"].notna().sum() if "avg_school_rating" in merged.columns else 0
        logger.info(f"  School ratings match: {matched}/{len(merged)}")

    # ── Walk Scores (ZIP-level, optional) ─────────────────────────────────
    walkscore = data.get("walkscore", pd.DataFrame())
    if not walkscore.empty and "zip_code" in walkscore.columns and "zip_code" in merged.columns:
        ws_cols = ["zip_code", "walk_score", "transit_score", "bike_score"]
        available = [c for c in ws_cols if c in walkscore.columns]
        merged = merged.merge(walkscore[available], on="zip_code", how="left")
        matched = merged["walk_score"].notna().sum() if "walk_score" in merged.columns else 0
        logger.info(f"  Walk score match: {matched}/{len(merged)}")

    # ── Neighborhoods (city + ZIP, optional) ──────────────────────────────
    nbr = data.get("neighborhoods", pd.DataFrame())
    if not nbr.empty:
        nbr = nbr.copy()
        nbr.columns = nbr.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        if "city" in nbr.columns:
            nbr["city"] = nbr["city"].str.strip().str.title()
        if "zip_code" in nbr.columns:
            nbr["zip_code"] = nbr["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
        # Only merge if it has data columns beyond keys
        nbr_data_cols = [c for c in nbr.columns if c not in ["city", "state", "zip_code", "source", "error"]]
        if nbr_data_cols and "zip_code" in nbr.columns and "zip_code" in merged.columns:
            # Avoid duplicating avg_school_rating if already merged from schools
            for col in nbr_data_cols:
                if col in merged.columns:
                    nbr = nbr.drop(columns=[col])
            nbr_data_cols = [c for c in nbr.columns if c not in ["city", "state", "zip_code", "source", "error"]]
            if nbr_data_cols:
                merge_cols = ["city", "zip_code"] + nbr_data_cols
                available = [c for c in merge_cols if c in nbr.columns]
                merged = merged.merge(nbr[available], on=["city", "zip_code"], how="left")
                logger.info(f"  Neighborhood merge: done")

    logger.info(f"\nFinal merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 5. QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame):
    logger.info(f"\n{'─'*60}")
    logger.info(f"DATA QUALITY REPORT")
    logger.info(f"{'─'*60}")
    logger.info(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")

    # Columns with data
    logger.info(f"\nColumns with data:")
    for col in df.columns:
        filled = df[col].notna().sum()
        pct = round(filled / len(df) * 100, 1)
        if pct > 0:
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            logger.info(f"  {col:<35} {filled:>5}/{len(df)}  ({pct:>5.1f}%) {bar}")

    # Numeric stats
    numeric = df.select_dtypes(include=[np.number])
    has_data = numeric.columns[numeric.notna().any()]
    if len(has_data) > 0:
        logger.info(f"\nKey numeric stats:")
        for col in has_data:
            vals = df[col].dropna()
            if len(vals) > 0:
                logger.info(f"  {col:<35} min={vals.min():>12,.1f}  median={vals.median():>12,.1f}  max={vals.max():>12,.1f}")

    # Records per city
    if "city" in df.columns:
        logger.info(f"\nRecords per city:")
        for city, count in df["city"].value_counts().items():
            logger.info(f"  {city}: {count}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("DATA CLEANING + MERGE PIPELINE")
    logger.info("=" * 60)

    # ── Load everything ───────────────────────────────────────────────────
    data = load_all_data()

    if data["properties"].empty:
        logger.error("No property data found. Run 01_data_collection.py first.")
        return

    # ── Clean properties ──────────────────────────────────────────────────
    properties, report = clean_properties(data["properties"])

    # ── Clean supplementary data ──────────────────────────────────────────
    logger.info("\nCleaning supplementary data...")
    clean_data = {
        "census": clean_census(data.get("census", pd.DataFrame())),
        "redfin_stats": clean_supplementary(data.get("redfin_stats", pd.DataFrame()), "Redfin stats"),
        "redfin_historical": clean_supplementary(data.get("redfin_historical", pd.DataFrame()), "Redfin historical"),
        "crime": clean_supplementary(data.get("crime", pd.DataFrame()), "Crime data"),
        "schools": clean_supplementary(data.get("schools", pd.DataFrame()), "School ratings"),
        "walkscore": clean_supplementary(data.get("walkscore", pd.DataFrame()), "Walk scores"),
        "neighborhoods": data.get("neighborhoods", pd.DataFrame()),
    }

    # ── Merge ─────────────────────────────────────────────────────────────
    merged = merge_all(properties, clean_data)

    # ── Save ──────────────────────────────────────────────────────────────
    merged.to_csv(CLEAN_DIR / "properties_cleaned.csv", index=False)
    logger.info(f"\nSaved: properties_cleaned.csv ({len(merged)} rows, {len(merged.columns)} cols)")

    # Also save cleaned supplementary files for reference
    for name, df in clean_data.items():
        if not df.empty:
            df.to_csv(CLEAN_DIR / f"{name}_cleaned.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────
    print_report(merged)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Properties:    {report['initial_rows']} → {report['final_rows']} rows")
    logger.info(f"  No price:    {report.get('dropped_no_price', 0)}")
    logger.info(f"  Outliers:    {report.get('price_outliers', 0)}")
    logger.info(f"  Duplicates:  {report.get('duplicates_removed', 0)}")
    logger.info(f"Final dataset: {len(merged)} rows × {len(merged.columns)} columns")
    logger.info(f"\n✅ Done! Next: python scripts/03_feature_engineering.py")


if __name__ == "__main__":
    main()