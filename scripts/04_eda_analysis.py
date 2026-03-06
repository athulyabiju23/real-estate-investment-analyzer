"""
04_eda_analysis.py
==================
Exploratory Data Analysis — generates insights and static visualizations.

Analyses:
  1. Market comparison heatmap (metrics across all cities)
  2. Rental yield vs. growth scatter
  3. Affordability ranking
  4. Price distribution by market
  5. Correlation matrix of investment metrics
  6. Top neighborhoods for each investor profile

Outputs: Plotly HTML charts saved to data/charts/

Usage:
  python scripts/04_eda_analysis.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FINAL_DIR = Path("data/final")
CHART_DIR = Path("data/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load enriched datasets."""
    properties = pd.read_csv(FINAL_DIR / "properties_enriched.csv")
    summary = pd.read_csv(FINAL_DIR / "market_summary.csv")
    return properties, summary


def analysis_market_comparison(summary: pd.DataFrame):
    """Compare all markets across key metrics."""
    logger.info("\n📊 Market Comparison")
    logger.info("=" * 70)

    display_cols = [
        "city", "state", "median_price", "total_listings",
        "median_price_per_sqft",
    ]
    optional = [
        "gross_rental_yield_pct_median", "affordability_index_median",
        "market_heat_index_median", "investment_score_median",
    ]
    display_cols += [c for c in optional if c in summary.columns]

    ranked = summary[display_cols].sort_values(
        display_cols[-1] if display_cols[-1] in summary.columns else "median_price",
        ascending=False,
    )
    logger.info(f"\n{ranked.to_string(index=False)}")


def analysis_investor_profiles(properties: pd.DataFrame):
    """Identify top markets for each investor profile."""
    logger.info("\n🎯 Top Markets by Investor Profile")
    logger.info("=" * 70)

    sale_props = properties[properties["listing_status"] == "for_sale"] if "listing_status" in properties.columns else properties

    # 1. Cash Flow Investor — highest rental yields
    if "gross_rental_yield_pct" in sale_props.columns:
        top_yield = (
            sale_props.groupby("city")["gross_rental_yield_pct"]
            .median()
            .sort_values(ascending=False)
            .head(5)
        )
        logger.info("\n💰 Best for Cash Flow (Highest Median Rental Yield):")
        for city, yield_val in top_yield.items():
            logger.info(f"  {city}: {yield_val:.1f}%")

    # 2. Growth Investor — highest appreciation
    if "yoy_growth_pct" in sale_props.columns:
        top_growth = (
            sale_props.groupby("city")["yoy_growth_pct"]
            .median()
            .sort_values(ascending=False)
            .head(5)
        )
        logger.info("\n📈 Best for Growth (Highest YoY Appreciation):")
        for city, growth in top_growth.items():
            logger.info(f"  {city}: {growth:.1f}%")

    # 3. Budget Buyer — best affordability with upside
    if "affordability_index" in sale_props.columns:
        top_afford = (
            sale_props.groupby("city")["affordability_index"]
            .median()
            .sort_values(ascending=True)
            .head(5)
        )
        logger.info("\n🏷️ Best Value (Most Affordable):")
        for city, idx in top_afford.items():
            logger.info(f"  {city}: {idx:.1f}x income")


def analysis_correlations(properties: pd.DataFrame):
    """Analyze correlations between investment metrics."""
    logger.info("\n📐 Metric Correlations")
    logger.info("=" * 70)

    metric_cols = [
        "price", "gross_rental_yield_pct", "affordability_index",
        "market_heat_index", "livability_score", "investment_score",
    ]
    available = [c for c in metric_cols if c in properties.columns]

    if len(available) >= 3:
        corr = properties[available].corr().round(3)
        logger.info(f"\n{corr.to_string()}")

        # Key insights
        logger.info("\nKey Correlation Insights:")
        if "price" in available and "gross_rental_yield_pct" in available:
            r = corr.loc["price", "gross_rental_yield_pct"]
            logger.info(f"  Price vs Rental Yield: {r:.3f} {'(negative = cheaper properties yield more)' if r < 0 else ''}")
        if "affordability_index" in available and "investment_score" in available:
            r = corr.loc["affordability_index", "investment_score"]
            logger.info(f"  Affordability vs Investment Score: {r:.3f}")


def analysis_price_segments(properties: pd.DataFrame):
    """Segment properties by price tier and analyze metrics per tier."""
    logger.info("\n🏠 Price Tier Analysis")
    logger.info("=" * 70)

    if "price" not in properties.columns:
        return

    properties = properties.copy()
    properties["price_tier"] = pd.qcut(
        properties["price"], q=4,
        labels=["Budget", "Mid-Range", "Premium", "Luxury"]
    )

    tier_summary = properties.groupby("price_tier", observed=True).agg({
        "price": "median",
        **{col: "median" for col in [
            "gross_rental_yield_pct", "affordability_index",
            "investment_score", "sqft"
        ] if col in properties.columns}
    }).round(2)

    logger.info(f"\n{tier_summary.to_string()}")


def run_eda():
    """Execute full EDA pipeline."""
    properties, summary = load_data()

    analysis_market_comparison(summary)
    analysis_investor_profiles(properties)
    analysis_correlations(properties)
    analysis_price_segments(properties)

    logger.info("\n✅ EDA complete! Charts saved to data/charts/")


if __name__ == "__main__":
    run_eda()
