"""
02_data_cleaning.py
Cleans scraped property data and merges with supplementary sources.
"""

import os, glob, logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW = Path("data/raw")
SUPP = Path("data/supplementary")
OUT = Path("data/cleaned")
OUT.mkdir(parents=True, exist_ok=True)

# ZIP prefixes to validate scraped zip codes
ZIP_PREFIX = {
    "Austin": ["787"], "Boise": ["837"], "Raleigh": ["276"],
    "Nashville": ["372"], "Phoenix": ["850"], "Tampa": ["336", "335"],
    "Charlotte": ["282"], "Salt Lake City": ["841"],
    "Denver": ["802"], "Jacksonville": ["322"],
    "New York": ["100", "101", "102", "103", "104"],
    "Boston": ["021"], "San Francisco": ["941"],
    "Los Angeles": ["900", "901", "902"], "Seattle": ["981"],
    "Chicago": ["606"], "Buffalo": ["142"], "Hartford": ["061"],
    "Durham": ["277"], "St. Louis": ["631"],
}


def find_latest(pattern, folder):
    files = sorted(glob.glob(str(folder / pattern)), key=os.path.getmtime, reverse=True)
    return Path(files[0]) if files else None


# ── Load all data ─────────────────────────────────────────────────────────────

def load_all():
    data = {}

    # Properties
    p = RAW / "properties_raw_latest.csv"
    if not p.exists():
        p = find_latest("properties_raw_*.csv", RAW)
    if p:
        data["props"] = pd.read_csv(p, dtype=str)
        log.info(f"Properties: {len(data['props'])} rows")

    # Census, market stats, crime, schools
    for name, pattern, folder in [
        ("census", "census_income_*.csv", RAW),
        ("redfin_stats", "redfin_market_stats_latest.csv", SUPP),
        ("redfin_hist", "redfin_historical_latest.csv", SUPP),
        ("crime", "crime_data_latest.csv", SUPP),
        ("schools", "school_ratings_latest.csv", SUPP),
    ]:
        f = find_latest(pattern, folder)
        if f:
            data[name] = pd.read_csv(f, dtype=str)
            log.info(f"{name}: {len(data[name])} rows")

    return data


# ── Clean properties ──────────────────────────────────────────────────────────

def clean_props(df):
    log.info(f"\nCleaning {len(df)} properties...")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Drop columns that are completely empty
    empty = [c for c in df.columns if df[c].dropna().astype(str).str.strip().isin(["", "nan", "None"]).all()
             or df[c].isna().all()]
    df = df.drop(columns=empty)
    log.info(f"Dropped {len(empty)} empty columns")

    # Convert numeric columns
    for col in ["price", "bathrooms", "price_per_sqft", "bedrooms", "sqft"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")

    # Standardize text
    if "city" in df.columns: df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns: df["state"] = df["state"].str.strip().str.upper()
    if "source" in df.columns: df["source"] = df["source"].str.strip().str.lower()

    # Fix zip codes — remove ones that don't match the city
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
        bad = 0
        for city, prefixes in ZIP_PREFIX.items():
            mask = (df["city"] == city) & df["zip_code"].notna() & ~df["zip_code"].str[:3].isin(prefixes)
            bad += mask.sum()
            df.loc[mask, "zip_code"] = pd.NA
        log.info(f"Fixed {bad} invalid zip codes")

    # Standardize listing status
    if "listing_status" in df.columns:
        mapping = {"for sale": "for_sale", "forsale": "for_sale", "for_sale": "for_sale",
                   "recently sold": "recently_sold", "recentlysold": "recently_sold",
                   "recently_sold": "recently_sold", "sold": "recently_sold"}
        df["listing_status"] = df["listing_status"].str.strip().str.lower().map(mapping).fillna("unknown")

    # Remove bad data
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 20_000) & (df["price"] <= 10_000_000)]
    if "sqft" in df.columns:
        df.loc[(df["sqft"] < 100) | (df["sqft"] > 15000), "sqft"] = np.nan
    if "bedrooms" in df.columns:
        df.loc[df["bedrooms"] > 10, "bedrooms"] = np.nan
    if "bathrooms" in df.columns:
        df.loc[df["bathrooms"] > 10, "bathrooms"] = np.nan

    # Fill missing sqft/beds/baths with city median
    for col in ["bedrooms", "bathrooms", "sqft"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("city")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # Deduplicate — keep redfin over realtor for same address
    if "full_address" in df.columns and "source" in df.columns:
        df["_rank"] = df["source"].map({"redfin": 0, "realtor": 1}).fillna(9)
        df = df.sort_values("_rank").drop_duplicates(subset=["full_address"], keep="first").drop(columns=["_rank"])

    # Recalculate price per sqft
    if "sqft" in df.columns:
        mask = df["sqft"] > 0
        df.loc[mask, "price_per_sqft"] = (df.loc[mask, "price"] / df.loc[mask, "sqft"]).round(2)

    df = df.reset_index(drop=True)
    log.info(f"Cleaned: {len(df)} properties")
    return df


# ── Clean supplementary data ─────────────────────────────────────────────────

def clean_supp(df, name):
    """Basic cleaning for any city-level supplementary CSV."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if "city" in df.columns: df["city"] = df["city"].str.strip().str.title()
    if "state" in df.columns: df["state"] = df["state"].str.strip().str.upper()
    # Cast everything that looks numeric
    for col in df.columns:
        if col not in ["city", "state", "source", "period_end", "school_density", "zip_code", "area_name"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop utility columns
    for col in ["source", "error", "url"]:
        if col in df.columns: df = df.drop(columns=[col])
    return df


def clean_census(df):
    df = clean_supp(df, "census")
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    # Census uses negative values for missing data
    for col in ["median_household_income", "population", "census_median_home_value", "census_median_rent"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df.drop_duplicates(subset=["zip_code"], keep="first")


# ── Merge everything ──────────────────────────────────────────────────────────

def merge_all(props, data):
    log.info("\nMerging datasets...")
    m = props.copy()

    # Census — join on zip, fall back to city median
    if "census" in data:
        census = data["census"]
        cols = [c for c in ["zip_code", "median_household_income", "population",
                            "census_median_home_value", "census_median_rent"] if c in census.columns]
        m = m.merge(census[cols], on="zip_code", how="left")
        matched = m["median_household_income"].notna().sum()
        log.info(f"  Census zip match: {matched}/{len(m)}")

        # City fallback for unmatched rows
        if matched < len(m) and "city" in census.columns:
            city_med = census.groupby("city").median(numeric_only=True).reset_index()
            for col in [c for c in cols if c != "zip_code"]:
                if col in city_med.columns:
                    city_map = city_med.set_index("city")[col].to_dict()
                    missing = m[col].isna()
                    m.loc[missing, col] = m.loc[missing, "city"].map(city_map)
            log.info(f"  Census after fallback: {m['median_household_income'].notna().sum()}/{len(m)}")

    # City-level joins — redfin stats, historical, crime, schools
    for key, label in [("redfin_stats", "Redfin stats"), ("redfin_hist", "Historical"),
                        ("crime", "Crime"), ("schools", "Schools")]:
        if key in data:
            supp = data[key]
            drop = [c for c in ["period_end", "latest_period"] if c in supp.columns]
            supp = supp.drop(columns=drop, errors="ignore")
            m = m.merge(supp, on=["city", "state"], how="left")
            log.info(f"  {label}: merged")

    log.info(f"Final: {len(m)} rows, {len(m.columns)} columns")
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("DATA CLEANING PIPELINE")
    log.info("=" * 50)

    data = load_all()
    if "props" not in data:
        log.error("No property data found!")
        return

    # Clean
    props = clean_props(data["props"])

    if "census" in data: data["census"] = clean_census(data["census"])
    for key in ["redfin_stats", "redfin_hist", "crime", "schools"]:
        if key in data: data[key] = clean_supp(data[key], key)

    # Merge
    merged = merge_all(props, data)

    # Save
    merged.to_csv(OUT / "properties_cleaned.csv", index=False)
    log.info(f"\nSaved: properties_cleaned.csv ({len(merged)} rows)")

    # Quick report
    log.info(f"\nColumns with data:")
    for col in merged.columns:
        pct = merged[col].notna().mean() * 100
        if pct > 0:
            log.info(f"  {col:<35} {pct:.0f}%")

    log.info("\n✅ Done! Next: python scripts/03_feature_engineering.py")


if __name__ == "__main__":
    main()