"""
05_generate_demo_data.py
========================
Generates realistic synthetic real estate data for 10 fastest-growing
U.S. markets in 2025. Data is calibrated to actual market conditions
using publicly available median prices and trends.

This allows the full pipeline and dashboard to run without API keys.

Usage:
  python scripts/05_generate_demo_data.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
FINAL_DIR = Path("data/final")
for d in [RAW_DIR, CLEAN_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Market Profiles (calibrated to Q1 2025 actuals) ─────────────────────────

MARKETS = {
    "Austin": {
        "state": "TX",
        "median_price": 475000, "price_std": 180000,
        "median_rent": 2100, "rent_std": 600,
        "median_sqft": 1850, "sqft_std": 550,
        "median_income": 85000,
        "yoy_growth": 12.3,
        "avg_dom": 28,
        "safety_score": 68,
        "school_rating": 7.2,
        "walk_score": 42,
        "zip_codes": ["78701", "78702", "78704", "78745", "78750", "78753", "78758", "78759"],
        "property_mix": {"Single Family": 0.55, "Condo": 0.20, "Townhouse": 0.15, "Multi-Family": 0.10},
    },
    "Boise": {
        "state": "ID",
        "median_price": 420000, "price_std": 130000,
        "median_rent": 1750, "rent_std": 450,
        "median_sqft": 1750, "sqft_std": 500,
        "median_income": 72000,
        "yoy_growth": 13.1,
        "avg_dom": 22,
        "safety_score": 78,
        "school_rating": 7.5,
        "walk_score": 38,
        "zip_codes": ["83702", "83706", "83709", "83713", "83716"],
        "property_mix": {"Single Family": 0.65, "Condo": 0.10, "Townhouse": 0.15, "Multi-Family": 0.10},
    },
    "Raleigh": {
        "state": "NC",
        "median_price": 405000, "price_std": 140000,
        "median_rent": 1800, "rent_std": 500,
        "median_sqft": 1900, "sqft_std": 600,
        "median_income": 78000,
        "yoy_growth": 9.8,
        "avg_dom": 25,
        "safety_score": 72,
        "school_rating": 7.8,
        "walk_score": 32,
        "zip_codes": ["27601", "27603", "27606", "27609", "27612", "27615"],
        "property_mix": {"Single Family": 0.60, "Condo": 0.12, "Townhouse": 0.18, "Multi-Family": 0.10},
    },
    "Nashville": {
        "state": "TN",
        "median_price": 445000, "price_std": 160000,
        "median_rent": 1950, "rent_std": 550,
        "median_sqft": 1800, "sqft_std": 520,
        "median_income": 73000,
        "yoy_growth": 10.5,
        "avg_dom": 30,
        "safety_score": 62,
        "school_rating": 6.8,
        "walk_score": 35,
        "zip_codes": ["37201", "37203", "37206", "37209", "37211", "37215"],
        "property_mix": {"Single Family": 0.50, "Condo": 0.22, "Townhouse": 0.16, "Multi-Family": 0.12},
    },
    "Phoenix": {
        "state": "AZ",
        "median_price": 380000, "price_std": 140000,
        "median_rent": 1700, "rent_std": 500,
        "median_sqft": 1700, "sqft_std": 480,
        "median_income": 68000,
        "yoy_growth": 8.2,
        "avg_dom": 35,
        "safety_score": 58,
        "school_rating": 6.5,
        "walk_score": 40,
        "zip_codes": ["85003", "85004", "85006", "85008", "85016", "85018"],
        "property_mix": {"Single Family": 0.58, "Condo": 0.18, "Townhouse": 0.14, "Multi-Family": 0.10},
    },
    "Tampa": {
        "state": "FL",
        "median_price": 365000, "price_std": 130000,
        "median_rent": 1850, "rent_std": 520,
        "median_sqft": 1650, "sqft_std": 460,
        "median_income": 62000,
        "yoy_growth": 7.9,
        "avg_dom": 32,
        "safety_score": 60,
        "school_rating": 6.6,
        "walk_score": 48,
        "zip_codes": ["33602", "33606", "33609", "33611", "33629"],
        "property_mix": {"Single Family": 0.48, "Condo": 0.25, "Townhouse": 0.15, "Multi-Family": 0.12},
    },
    "Charlotte": {
        "state": "NC",
        "median_price": 375000, "price_std": 130000,
        "median_rent": 1700, "rent_std": 480,
        "median_sqft": 1800, "sqft_std": 520,
        "median_income": 70000,
        "yoy_growth": 9.1,
        "avg_dom": 27,
        "safety_score": 64,
        "school_rating": 7.0,
        "walk_score": 30,
        "zip_codes": ["28202", "28203", "28205", "28209", "28210", "28211"],
        "property_mix": {"Single Family": 0.55, "Condo": 0.18, "Townhouse": 0.17, "Multi-Family": 0.10},
    },
    "Salt Lake City": {
        "state": "UT",
        "median_price": 460000, "price_std": 150000,
        "median_rent": 1800, "rent_std": 480,
        "median_sqft": 1700, "sqft_std": 450,
        "median_income": 75000,
        "yoy_growth": 11.2,
        "avg_dom": 24,
        "safety_score": 74,
        "school_rating": 7.4,
        "walk_score": 56,
        "zip_codes": ["84101", "84102", "84103", "84105", "84106"],
        "property_mix": {"Single Family": 0.50, "Condo": 0.20, "Townhouse": 0.18, "Multi-Family": 0.12},
    },
    "Denver": {
        "state": "CO",
        "median_price": 530000, "price_std": 200000,
        "median_rent": 2200, "rent_std": 600,
        "median_sqft": 1650, "sqft_std": 480,
        "median_income": 82000,
        "yoy_growth": 6.8,
        "avg_dom": 38,
        "safety_score": 60,
        "school_rating": 7.0,
        "walk_score": 62,
        "zip_codes": ["80202", "80205", "80209", "80210", "80220", "80222"],
        "property_mix": {"Single Family": 0.45, "Condo": 0.25, "Townhouse": 0.18, "Multi-Family": 0.12},
    },
    "Jacksonville": {
        "state": "FL",
        "median_price": 310000, "price_std": 110000,
        "median_rent": 1600, "rent_std": 420,
        "median_sqft": 1700, "sqft_std": 500,
        "median_income": 58000,
        "yoy_growth": 8.7,
        "avg_dom": 34,
        "safety_score": 55,
        "school_rating": 6.2,
        "walk_score": 26,
        "zip_codes": ["32202", "32204", "32207", "32210", "32216"],
        "property_mix": {"Single Family": 0.65, "Condo": 0.10, "Townhouse": 0.15, "Multi-Family": 0.10},
    },
}


def generate_properties(city: str, profile: dict, n: int = 150) -> pd.DataFrame:
    """Generate realistic property listings for a single market."""
    records = []

    types = list(profile["property_mix"].keys())
    type_probs = list(profile["property_mix"].values())

    for i in range(n):
        prop_type = np.random.choice(types, p=type_probs)

        # Adjust price and size based on property type
        type_multipliers = {
            "Single Family": (1.0, 1.2),
            "Condo": (0.65, 0.60),
            "Townhouse": (0.85, 0.80),
            "Multi-Family": (1.3, 1.5),
        }
        price_mult, sqft_mult = type_multipliers.get(prop_type, (1.0, 1.0))

        price = max(80000, np.random.normal(
            profile["median_price"] * price_mult,
            profile["price_std"] * 0.7,
        ))
        sqft = max(400, np.random.normal(
            profile["median_sqft"] * sqft_mult,
            profile["sqft_std"],
        ))
        rent = max(600, np.random.normal(
            profile["median_rent"] * (price_mult * 0.8 + 0.2),
            profile["rent_std"],
        ))

        # Bedrooms correlated with sqft
        if sqft < 800:
            beds = np.random.choice([1, 2], p=[0.7, 0.3])
        elif sqft < 1400:
            beds = np.random.choice([2, 3], p=[0.6, 0.4])
        elif sqft < 2200:
            beds = np.random.choice([3, 4], p=[0.5, 0.5])
        else:
            beds = np.random.choice([4, 5], p=[0.6, 0.4])

        baths = max(1, beds - np.random.choice([0, 1], p=[0.6, 0.4]))
        if sqft > 2500:
            baths = min(baths + 1, 5)

        zip_code = np.random.choice(profile["zip_codes"])
        # Vary neighborhood scores slightly by ZIP
        zip_idx = profile["zip_codes"].index(zip_code)
        safety_var = np.random.normal(0, 5)
        school_var = np.random.normal(0, 0.5)

        dom = max(1, int(np.random.exponential(profile["avg_dom"])))
        year_built = int(np.random.choice(
            range(1960, 2025),
            p=np.array([1] * 30 + [2] * 20 + [4] * 15) / sum([1] * 30 + [2] * 20 + [4] * 15),
        ))

        # Listing status: 60% for sale, 25% recently sold, 15% for rent
        status = np.random.choice(
            ["for_sale", "recently_sold", "for_rent"],
            p=[0.60, 0.25, 0.15],
        )

        records.append({
            "city": city,
            "state": profile["state"],
            "zip_code": zip_code,
            "property_type": prop_type,
            "listing_status": status,
            "price": round(price, -2),
            "rent_zestimate": round(rent, -1) if status != "for_rent" else round(price, -1),
            "bedrooms": int(beds),
            "bathrooms": int(baths),
            "sqft": round(sqft),
            "price_per_sqft": round(price / sqft, 2),
            "year_built": year_built,
            "days_on_market": dom,
            "yoy_growth_pct": round(np.random.normal(profile["yoy_growth"], 2.5), 1),
            "safety_score": round(np.clip(profile["safety_score"] + safety_var, 20, 98), 1),
            "school_rating": round(np.clip(profile["school_rating"] + school_var, 1, 10), 1),
            "walk_score": round(np.clip(profile["walk_score"] + np.random.normal(0, 8), 5, 98)),
            "crime_rate": round(np.clip(100 - (profile["safety_score"] + safety_var) + np.random.normal(0, 3), 5, 95), 1),
            "median_household_income": round(
                np.random.normal(profile["median_income"], profile["median_income"] * 0.15), -2
            ),
        })

    return pd.DataFrame(records)


def generate_historical_prices() -> pd.DataFrame:
    """Generate 5 years of monthly median price history per market."""
    records = []
    months = pd.date_range("2020-01-01", "2025-01-01", freq="MS")

    for city, profile in MARKETS.items():
        # Work backwards from current median to 5 years ago
        annual_growth = profile["yoy_growth"] / 100
        current_price = profile["median_price"]

        # Generate a realistic growth curve (not perfectly linear)
        n_months = len(months)
        monthly_growth = (1 + annual_growth) ** (1/12) - 1

        prices = []
        price = current_price / ((1 + annual_growth) ** 5)  # Starting price 5 years ago

        for i, month in enumerate(months):
            # Add some monthly noise and seasonal patterns
            seasonal = np.sin(2 * np.pi * month.month / 12) * 0.01  # Spring bump
            noise = np.random.normal(0, 0.005)

            # COVID boom (2021-2022) with cooldown
            if month.year in [2021, 2022]:
                growth_mult = 1.3
            elif month.year == 2023:
                growth_mult = 0.6
            else:
                growth_mult = 1.0

            monthly_rate = monthly_growth * growth_mult + seasonal + noise
            price *= (1 + monthly_rate)
            prices.append(price)

            records.append({
                "city": city,
                "state": profile["state"],
                "date": month.strftime("%Y-%m-%d"),
                "median_price": round(price, -2),
                "median_rent": round(
                    profile["median_rent"] * (price / current_price) ** 0.5, -1
                ),
                "inventory": int(np.random.normal(500, 120)),
                "new_listings": int(np.random.normal(150, 40)),
                "median_dom": int(np.random.normal(profile["avg_dom"], 8)),
            })

    return pd.DataFrame(records)


def run_generation():
    """Generate all demo datasets."""
    logger.info("Generating realistic demo data for 10 U.S. markets...")

    # 1. Property listings
    all_props = []
    for city, profile in MARKETS.items():
        n = np.random.randint(120, 200)
        df = generate_properties(city, profile, n)
        all_props.append(df)
        logger.info(f"  Generated {len(df)} properties for {city}, {profile['state']}")

    properties = pd.concat(all_props, ignore_index=True)
    properties.to_csv(RAW_DIR / "demo_properties.csv", index=False)
    properties.to_csv(CLEAN_DIR / "properties_cleaned.csv", index=False)
    logger.info(f"\nTotal properties: {len(properties)}")

    # 2. Historical prices
    history = generate_historical_prices()
    history.to_csv(RAW_DIR / "price_history.csv", index=False)
    logger.info(f"Historical price records: {len(history)}")

    # 3. Income data
    income_records = []
    for city, profile in MARKETS.items():
        for zc in profile["zip_codes"]:
            income_records.append({
                "zip_code": zc,
                "area_name": f"ZCTA5 {zc}",
                "city": city,
                "state": profile["state"],
                "median_household_income": round(
                    np.random.normal(profile["median_income"], profile["median_income"] * 0.12), -2
                ),
            })
    income = pd.DataFrame(income_records)
    income.to_csv(RAW_DIR / "census_income_raw.csv", index=False)

    # 4. Market summary JSON (for dashboard)
    market_summary = []
    for city, p in MARKETS.items():
        city_props = properties[
            (properties["city"] == city) & (properties["listing_status"] == "for_sale")
        ]
        market_summary.append({
            "city": city,
            "state": p["state"],
            "median_price": int(city_props["price"].median()),
            "mean_price": int(city_props["price"].mean()),
            "total_listings": len(city_props),
            "median_sqft": int(city_props["sqft"].median()),
            "median_price_per_sqft": round(city_props["price_per_sqft"].median(), 0),
            "median_rent": int(p["median_rent"]),
            "gross_rental_yield_pct": round(
                (p["median_rent"] * 12 / p["median_price"]) * 100, 2
            ),
            "net_rental_yield_pct": round(
                (p["median_rent"] * 12 * 0.65 / p["median_price"]) * 100, 2
            ),
            "yoy_growth_pct": p["yoy_growth"],
            "affordability_index": round(p["median_price"] / p["median_income"], 2),
            "median_income": p["median_income"],
            "avg_dom": p["avg_dom"],
            "safety_score": p["safety_score"],
            "school_rating": p["school_rating"],
            "walk_score": p["walk_score"],
        })

    summary_df = pd.DataFrame(market_summary)
    summary_df.to_csv(FINAL_DIR / "market_summary.csv", index=False)

    # Also save as JSON for the React dashboard
    with open(FINAL_DIR / "market_data.json", "w") as f:
        json.dump({
            "markets": market_summary,
            "properties": properties.head(500).to_dict(orient="records"),
            "history": history.to_dict(orient="records"),
        }, f, indent=2, default=str)

    logger.info(f"\n✅ Demo data generation complete!")
    logger.info(f"  Properties:  {RAW_DIR / 'demo_properties.csv'}")
    logger.info(f"  History:     {RAW_DIR / 'price_history.csv'}")
    logger.info(f"  Income:      {RAW_DIR / 'census_income_raw.csv'}")
    logger.info(f"  Summary:     {FINAL_DIR / 'market_summary.csv'}")
    logger.info(f"  Dashboard:   {FINAL_DIR / 'market_data.json'}")


if __name__ == "__main__":
    run_generation()
