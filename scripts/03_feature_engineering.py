"""
03_feature_engineering.py
=========================
Calculates investment metrics from the cleaned, merged dataset.

Input:  data/cleaned/properties_cleaned.csv (608 rows, 41 columns)
Output: data/final/properties_enriched.csv   (608 rows, ~55 columns)
        data/final/market_summary.csv        (10 rows, city-level aggregates)

Metrics Calculated:
  1. Rental Yield (gross + net) — using census_median_rent
  2. Price-to-Rent Ratio
  3. Affordability Index + category + mortgage estimate
  4. Market Heat Index (composite: DOM, sale-to-list, sold above list, inventory)
  5. Safety Index (already have safety_score, normalize it)
  6. Livability Score (composite: safety, schools, market desirability)
  7. Investment Score (composite: yield, affordability, growth, heat, livability)
  8. Price vs Market comparison

Usage:
  python scripts/03_feature_engineering.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CLEAN_DIR = Path("data/cleaned")
FINAL_DIR = Path("data/final")
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helper ──────────────────────────────────────────────────────────────────

def min_max_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize a series to 0–100 scale."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(50.0, index=series.index)
    normalized = (series - min_val) / (max_val - min_val) * 100
    return normalized if higher_is_better else 100 - normalized


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    path = CLEAN_DIR / "properties_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 02_data_cleaning.py first. Missing: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RENTAL YIELD
# ═══════════════════════════════════════════════════════════════════════════════

def calc_rental_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gross Rental Yield = (census_median_rent × 12 / price) × 100
    Net Rental Yield   = (census_median_rent × 12 × 0.65 / price) × 100
      → 35% expense ratio: taxes, insurance, maintenance, vacancy
    Price-to-Rent Ratio = price / (annual_rent)
      → Under 15 = buy favored, Over 20 = rent favored
    """
    df = df.copy()

    if "census_median_rent" in df.columns and "price" in df.columns:
        annual_rent = df["census_median_rent"] * 12
        valid = (df["price"] > 0) & (df["census_median_rent"] > 0)

        df["gross_rental_yield_pct"] = np.where(
            valid, (annual_rent / df["price"] * 100).round(2), np.nan
        )

        df["net_rental_yield_pct"] = np.where(
            valid, (annual_rent * 0.65 / df["price"] * 100).round(2), np.nan
        )

        df["price_to_rent_ratio"] = np.where(
            valid & (annual_rent > 0), (df["price"] / annual_rent).round(1), np.nan
        )

        df["monthly_rent_estimate"] = df["census_median_rent"]

        logger.info(f"Rental yield: median gross = {df['gross_rental_yield_pct'].median():.2f}%")
    else:
        logger.warning("Missing census_median_rent or price — skipping rental yield")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AFFORDABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def calc_affordability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Affordability Index = price / median_household_income
      < 3.0  → Very Affordable
      3-4    → Affordable
      4-5    → Moderately Unaffordable
      5-7    → Seriously Unaffordable
      > 7    → Severely Unaffordable

    Also estimates monthly mortgage payment and % of income.
    """
    df = df.copy()

    if "median_household_income" in df.columns and "price" in df.columns:
        valid = (df["median_household_income"] > 0) & (df["price"] > 0)

        df["affordability_index"] = np.where(
            valid, (df["price"] / df["median_household_income"]).round(2), np.nan
        )

        bins = [0, 3, 4, 5, 7, float("inf")]
        labels = ["Very Affordable", "Affordable", "Moderately Unaffordable",
                   "Seriously Unaffordable", "Severely Unaffordable"]
        df["affordability_category"] = pd.cut(
            df["affordability_index"], bins=bins, labels=labels
        )

        # Monthly mortgage estimate: 30yr fixed at 6.5%, 20% down
        rate = 0.065 / 12
        n = 360
        loan = df["price"] * 0.80
        monthly = loan * (rate * (1 + rate)**n) / ((1 + rate)**n - 1)
        df["est_monthly_mortgage"] = monthly.round(0)

        df["mortgage_pct_of_income"] = np.where(
            valid,
            (monthly * 12 / df["median_household_income"] * 100).round(1),
            np.nan
        )

        logger.info(f"Affordability: median index = {df['affordability_index'].median():.2f}")
    else:
        logger.warning("Missing income or price — skipping affordability")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MARKET HEAT INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def calc_market_heat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market Heat Index (0–100): how competitive is this market?

    Components (all from Redfin Data Center):
      - Median DOM (inverse: fewer days = hotter)        → 30%
      - Avg sale-to-list ratio (higher = hotter)          → 25%
      - % sold above list (higher = hotter)               → 20%
      - % off market in 2 weeks (higher = hotter)         → 15%
      - Months of supply (inverse: lower = hotter)        → 10%
    """
    df = df.copy()

    components = {}

    # DOM score: fewer days = hotter
    if "median_dom" in df.columns:
        max_dom = df["median_dom"].quantile(0.95)
        if max_dom > 0:
            components["dom_score"] = ((1 - df["median_dom"].clip(upper=max_dom) / max_dom) * 100).clip(0, 100)

    # Sale-to-list: higher = hotter (values around 0.95–1.05)
    if "avg_sale_to_list" in df.columns:
        components["stl_score"] = ((df["avg_sale_to_list"] - 0.90) / 0.15 * 100).clip(0, 100)

    # Sold above list: higher = hotter (values 0–0.5 typically)
    if "sold_above_list" in df.columns:
        components["above_list_score"] = (df["sold_above_list"] * 200).clip(0, 100)

    # Off market in 2 weeks: higher = hotter
    if "off_market_in_two_weeks" in df.columns:
        components["fast_sale_score"] = (df["off_market_in_two_weeks"] * 100).clip(0, 100)

    # Months of supply: lower = hotter (balanced market = 4-6 months)
    if "months_of_supply" in df.columns:
        max_mos = df["months_of_supply"].quantile(0.95)
        if max_mos > 0:
            components["supply_score"] = ((1 - df["months_of_supply"].clip(upper=max_mos) / max_mos) * 100).clip(0, 100)

    if components:
        weights = {
            "dom_score": 0.30,
            "stl_score": 0.25,
            "above_list_score": 0.20,
            "fast_sale_score": 0.15,
            "supply_score": 0.10,
        }

        total_weight = sum(weights[k] for k in components if k in weights)
        df["market_heat_index"] = sum(
            components[k] * weights.get(k, 0) / total_weight
            for k in components
        ).round(1)

        logger.info(f"Market heat: median = {df['market_heat_index'].median():.1f}")
    else:
        logger.warning("No market data available for heat index")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LIVABILITY SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_livability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Livability Score (0–100): quality of life composite.

    Components:
      - Safety score (from crime data)          → 35%
      - School rating (scaled to 0–100)          → 30%
      - Market desirability (from market heat)   → 20%
      - Affordability (inverse index)            → 15%
    """
    df = df.copy()

    components = {}

    # Safety: already 0–100 from crime collector
    if "safety_score" in df.columns:
        components["safety"] = df["safety_score"]

    # Schools: 1–10 scale → 0–100
    if "avg_school_rating" in df.columns:
        components["schools"] = (df["avg_school_rating"] * 10).clip(0, 100)

    # Market desirability (proxy from heat — moderate is best for livability)
    if "market_heat_index" in df.columns:
        components["desirability"] = df["market_heat_index"]

    # Affordability component (lower index = more livable)
    if "affordability_index" in df.columns:
        components["afford"] = min_max_normalize(df["affordability_index"], higher_is_better=False)

    if components:
        weights = {"safety": 0.35, "schools": 0.30, "desirability": 0.20, "afford": 0.15}
        total_weight = sum(weights[k] for k in components if k in weights)

        df["livability_score"] = sum(
            components[k] * weights.get(k, 0) / total_weight
            for k in components
        ).round(1)

        logger.info(f"Livability: median = {df['livability_score'].median():.1f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRICE VS MARKET
# ═══════════════════════════════════════════════════════════════════════════════

def calc_price_vs_market(df: pd.DataFrame) -> pd.DataFrame:
    """How does this property compare to its city's median?"""
    df = df.copy()

    if "price" in df.columns and "median_sale_price" in df.columns:
        valid = df["median_sale_price"] > 0
        df["pct_vs_market"] = np.where(
            valid,
            (((df["price"] - df["median_sale_price"]) / df["median_sale_price"]) * 100).round(1),
            np.nan
        )

        df["price_category"] = pd.cut(
            df["pct_vs_market"],
            bins=[-float("inf"), -20, -5, 5, 20, float("inf")],
            labels=["Well Below Market", "Below Market", "At Market", "Above Market", "Well Above Market"]
        )

        logger.info(f"Price vs market: median = {df['pct_vs_market'].median():.1f}%")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INVESTMENT SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_investment_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Investment Score (0–100): overall investment attractiveness.

    Components:
      - Rental yield (higher = better)           → 30%
      - Affordability (lower index = better)      → 20%
      - YoY growth (higher = better)              → 20%
      - Market heat (moderate ~50 is ideal)        → 15%
      - Livability (higher = better)               → 15%
    """
    df = df.copy()

    components = {}

    if "gross_rental_yield_pct" in df.columns:
        components["yield"] = min_max_normalize(df["gross_rental_yield_pct"], True)

    if "affordability_index" in df.columns:
        components["afford"] = min_max_normalize(df["affordability_index"], False)

    if "yoy_price_change_pct" in df.columns:
        components["growth"] = min_max_normalize(df["yoy_price_change_pct"], True)

    if "market_heat_index" in df.columns:
        # Moderate heat is best — penalize extremes
        heat = df["market_heat_index"]
        ideal = 55
        components["heat"] = (100 - (abs(heat - ideal) * 1.5)).clip(0, 100)

    if "livability_score" in df.columns:
        components["livability"] = min_max_normalize(df["livability_score"], True)

    if components:
        weights = {"yield": 0.30, "afford": 0.20, "growth": 0.20, "heat": 0.15, "livability": 0.15}
        total_weight = sum(weights[k] for k in components if k in weights)

        df["investment_score"] = sum(
            components[k] * weights.get(k, 0) / total_weight
            for k in components
        ).round(1)

        # Percentile rank
        df["investment_percentile"] = (df["investment_score"].rank(pct=True) * 100).round(1)

        # Investment tier
        df["investment_tier"] = pd.cut(
            df["investment_percentile"],
            bins=[0, 25, 50, 75, 100],
            labels=["Avoid", "Below Average", "Good", "Excellent"],
            include_lowest=True,
        )

        logger.info(f"Investment score: median = {df['investment_score'].median():.1f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MARKET SUMMARY (city-level aggregation)
# ═══════════════════════════════════════════════════════════════════════════════

def create_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per city for dashboard."""

    agg = {
        "price": ["median", "mean", "min", "max", "count"],
        "sqft": ["median"],
        "price_per_sqft": ["median"],
        "bedrooms": ["median"],
    }

    # Add all calculated metrics
    optional = [
        "gross_rental_yield_pct", "net_rental_yield_pct", "price_to_rent_ratio",
        "affordability_index", "est_monthly_mortgage", "mortgage_pct_of_income",
        "market_heat_index", "livability_score", "investment_score",
        "safety_score", "avg_school_rating",
        "median_dom", "avg_sale_to_list", "inventory", "months_of_supply",
        "yoy_price_change_pct", "median_sale_price",
        "median_household_income", "census_median_rent",
    ]
    for col in optional:
        if col in df.columns:
            agg[col] = ["median"]

    summary = df.groupby(["city", "state"]).agg(agg).round(2)
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary = summary.reset_index()

    # Clean up column names
    rename = {
        "price_median": "median_price",
        "price_mean": "avg_price",
        "price_min": "min_price",
        "price_max": "max_price",
        "price_count": "total_listings",
        "sqft_median": "median_sqft",
        "price_per_sqft_median": "median_ppsf",
        "bedrooms_median": "median_beds",
    }
    for old, new in rename.items():
        if old in summary.columns:
            summary = summary.rename(columns={old: new})

    # Also rename the _median suffix columns
    for col in summary.columns:
        if col.endswith("_median") and col not in rename:
            new_name = col.replace("_median", "")
            if new_name not in summary.columns:
                summary = summary.rename(columns={col: new_name})

    # Sort by investment score
    if "investment_score" in summary.columns:
        summary = summary.sort_values("investment_score", ascending=False)

    summary = summary.reset_index(drop=True)
    logger.info(f"Market summary: {len(summary)} cities")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Load
    df = load_data()

    # Calculate all metrics
    df = calc_rental_yield(df)
    df = calc_affordability(df)
    df = calc_market_heat(df)
    df = calc_livability(df)
    df = calc_price_vs_market(df)
    df = calc_investment_score(df)

    # Save enriched property data
    df.to_csv(FINAL_DIR / "properties_enriched.csv", index=False)
    logger.info(f"\nSaved: properties_enriched.csv ({len(df)} rows, {len(df.columns)} columns)")

    # Create and save market summary
    summary = create_market_summary(df)
    summary.to_csv(FINAL_DIR / "market_summary.csv", index=False)
    logger.info(f"Saved: market_summary.csv ({len(summary)} rows)")

    # ── Print Results ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"\nNew columns added:")
    new_cols = [
        "gross_rental_yield_pct", "net_rental_yield_pct", "price_to_rent_ratio",
        "monthly_rent_estimate", "affordability_index", "affordability_category",
        "est_monthly_mortgage", "mortgage_pct_of_income",
        "market_heat_index", "livability_score", "pct_vs_market", "price_category",
        "investment_score", "investment_percentile", "investment_tier",
    ]
    for col in new_cols:
        if col in df.columns:
            logger.info(f"  ✓ {col}")

    # Market comparison
    logger.info(f"\n{'─'*60}")
    logger.info("MARKET COMPARISON (sorted by investment score)")
    logger.info(f"{'─'*60}")
    display_cols = ["city", "state"]
    for col in ["total_listings", "median_price", "gross_rental_yield_pct",
                 "yoy_price_change_pct", "market_heat_index", "safety_score",
                 "livability_score", "investment_score"]:
        if col in summary.columns:
            display_cols.append(col)
    logger.info(f"\n{summary[display_cols].to_string(index=False)}")

    # Investor profiles
    logger.info(f"\n{'─'*60}")
    logger.info("TOP PICKS BY INVESTOR PROFILE")
    logger.info(f"{'─'*60}")

    if "gross_rental_yield_pct" in summary.columns:
        top_yield = summary.nlargest(3, "gross_rental_yield_pct")
        logger.info("\n💰 Best for Cash Flow (highest rental yield):")
        for _, r in top_yield.iterrows():
            logger.info(f"  {r['city']}, {r['state']}: {r['gross_rental_yield_pct']:.1f}% yield, ${r.get('median_price', 0):,.0f} median")

    if "yoy_price_change_pct" in summary.columns:
        top_growth = summary.nlargest(3, "yoy_price_change_pct")
        logger.info("\n📈 Best for Growth (highest YoY appreciation):")
        for _, r in top_growth.iterrows():
            logger.info(f"  {r['city']}, {r['state']}: {r['yoy_price_change_pct']:.1f}% YoY, ${r.get('median_price', 0):,.0f} median")

    if "investment_score" in summary.columns and "median_price" in summary.columns:
        budget = summary[summary["median_price"] <= 400000].nlargest(3, "investment_score")
        if not budget.empty:
            logger.info("\n🏷️ Best Budget Picks (under $400K, highest investment score):")
            for _, r in budget.iterrows():
                logger.info(f"  {r['city']}, {r['state']}: score {r['investment_score']:.1f}, ${r['median_price']:,.0f} median")

    if "livability_score" in summary.columns:
        top_live = summary.nlargest(3, "livability_score")
        logger.info("\n🏡 Best for Living (highest livability):")
        for _, r in top_live.iterrows():
            logger.info(f"  {r['city']}, {r['state']}: livability {r['livability_score']:.1f}, safety {r.get('safety_score', 0):.1f}")

    logger.info(f"\n✅ Feature engineering complete!")
    logger.info(f"Next: build your dashboard from data/final/market_summary.csv")


if __name__ == "__main__":
    main()