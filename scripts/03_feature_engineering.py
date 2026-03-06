"""
03_feature_engineering.py
Calculates investment metrics from cleaned property data.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IN = Path("data/cleaned")
OUT = Path("data/final")
OUT.mkdir(parents=True, exist_ok=True)


def normalize(s, higher_better=True):
    """Min-max scale to 0-100."""
    if s.max() == s.min(): return pd.Series(50, index=s.index)
    n = (s - s.min()) / (s.max() - s.min()) * 100
    return n if higher_better else 100 - n


def main():
    log.info("=" * 50)
    log.info("FEATURE ENGINEERING")
    log.info("=" * 50)

    df = pd.read_csv(IN / "properties_cleaned.csv")
    log.info(f"Loaded {len(df)} rows")

    # ── Rental Yield ──────────────────────────────────────────────────────
    # Gross = (monthly rent × 12 / price) × 100
    # Net = gross × 0.65 (35% expense ratio for taxes, insurance, maintenance)
    if "census_median_rent" in df.columns:
        annual_rent = df["census_median_rent"] * 12
        valid = (df["price"] > 0) & (df["census_median_rent"] > 0)

        df["gross_rental_yield_pct"] = np.where(valid, (annual_rent / df["price"] * 100).round(2), np.nan)
        df["net_rental_yield_pct"] = np.where(valid, (annual_rent * 0.65 / df["price"] * 100).round(2), np.nan)
        df["price_to_rent_ratio"] = np.where(valid & (annual_rent > 0), (df["price"] / annual_rent).round(1), np.nan)
        df["monthly_rent_estimate"] = df["census_median_rent"]

        log.info(f"Rental yield: median {df['gross_rental_yield_pct'].median():.2f}%")

    # ── Affordability ─────────────────────────────────────────────────────
    # Index = price / income. Under 3 = affordable, over 7 = severely unaffordable
    if "median_household_income" in df.columns:
        valid = df["median_household_income"] > 0
        df["affordability_index"] = np.where(valid, (df["price"] / df["median_household_income"]).round(2), np.nan)

        df["affordability_category"] = pd.cut(
            df["affordability_index"],
            bins=[0, 3, 4, 5, 7, float("inf")],
            labels=["Very Affordable", "Affordable", "Moderate", "Serious", "Severe"]
        )

        # Monthly mortgage: 30yr fixed, 6.5%, 20% down
        rate = 0.065 / 12
        loan = df["price"] * 0.80
        monthly = loan * (rate * (1 + rate)**360) / ((1 + rate)**360 - 1)
        df["est_monthly_mortgage"] = monthly.round(0)
        df["mortgage_pct_of_income"] = np.where(valid, (monthly * 12 / df["median_household_income"] * 100).round(1), np.nan)

        log.info(f"Affordability: median index {df['affordability_index'].median():.2f}")

    # ── Market Heat ───────────────────────────────────────────────────────
    # Composite of DOM, sale-to-list ratio, sold above list, supply
    components = {}
    if "median_dom" in df.columns:
        mx = df["median_dom"].quantile(0.95)
        if mx > 0: components["dom"] = ((1 - df["median_dom"].clip(upper=mx) / mx) * 100).clip(0, 100)
    if "avg_sale_to_list" in df.columns:
        components["stl"] = ((df["avg_sale_to_list"] - 0.90) / 0.15 * 100).clip(0, 100)
    if "sold_above_list" in df.columns:
        components["above"] = (df["sold_above_list"] * 200).clip(0, 100)
    if "months_of_supply" in df.columns:
        mx = df["months_of_supply"].quantile(0.95)
        if mx > 0: components["supply"] = ((1 - df["months_of_supply"].clip(upper=mx) / mx) * 100).clip(0, 100)

    if components:
        weights = {"dom": 0.35, "stl": 0.25, "above": 0.20, "supply": 0.20}
        total_w = sum(weights.get(k, 0) for k in components)
        df["market_heat_index"] = sum(components[k] * weights.get(k, 0) / total_w for k in components).round(1)
        log.info(f"Market heat: median {df['market_heat_index'].median():.1f}")

    # ── Livability ────────────────────────────────────────────────────────
    # Weighted: safety 35%, schools 30%, market desirability 20%, affordability 15%
    parts = {}
    if "safety_score" in df.columns: parts["safety"] = df["safety_score"]
    if "avg_school_rating" in df.columns: parts["schools"] = df["avg_school_rating"] * 10
    if "market_heat_index" in df.columns: parts["heat"] = df["market_heat_index"]
    if "affordability_index" in df.columns: parts["afford"] = normalize(df["affordability_index"], higher_better=False)

    if parts:
        w = {"safety": 0.35, "schools": 0.30, "heat": 0.20, "afford": 0.15}
        tw = sum(w.get(k, 0) for k in parts)
        df["livability_score"] = sum(parts[k] * w.get(k, 0) / tw for k in parts).round(1)
        log.info(f"Livability: median {df['livability_score'].median():.1f}")

    # ── Price vs Market ───────────────────────────────────────────────────
    if "median_sale_price" in df.columns:
        valid = df["median_sale_price"] > 0
        df["pct_vs_market"] = np.where(valid,
            ((df["price"] - df["median_sale_price"]) / df["median_sale_price"] * 100).round(1), np.nan)
        df["price_category"] = pd.cut(df["pct_vs_market"],
            bins=[-np.inf, -20, -5, 5, 20, np.inf],
            labels=["Well Below", "Below", "At Market", "Above", "Well Above"])

    # ── Investment Score ──────────────────────────────────────────────────
    # Weighted composite: yield 30%, affordability 20%, growth 20%, heat 15%, livability 15%
    parts = {}
    if "gross_rental_yield_pct" in df.columns: parts["yield"] = normalize(df["gross_rental_yield_pct"])
    if "affordability_index" in df.columns: parts["afford"] = normalize(df["affordability_index"], False)
    if "yoy_price_change_pct" in df.columns: parts["growth"] = normalize(df["yoy_price_change_pct"])
    if "market_heat_index" in df.columns: parts["heat"] = (100 - (abs(df["market_heat_index"] - 55) * 1.5)).clip(0, 100)
    if "livability_score" in df.columns: parts["live"] = normalize(df["livability_score"])

    if parts:
        w = {"yield": 0.30, "afford": 0.20, "growth": 0.20, "heat": 0.15, "live": 0.15}
        tw = sum(w.get(k, 0) for k in parts)
        df["investment_score"] = sum(parts[k] * w.get(k, 0) / tw for k in parts).round(1)
        df["investment_percentile"] = (df["investment_score"].rank(pct=True) * 100).round(1)
        df["investment_tier"] = pd.cut(df["investment_percentile"],
            bins=[0, 25, 50, 75, 100], labels=["Avoid", "Below Avg", "Good", "Excellent"], include_lowest=True)
        log.info(f"Investment score: median {df['investment_score'].median():.1f}")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(OUT / "properties_enriched.csv", index=False)
    log.info(f"\nSaved: properties_enriched.csv ({len(df)} rows, {len(df.columns)} cols)")

    # ── Market summary ────────────────────────────────────────────────────
    agg = {"price": ["median", "mean", "min", "max", "count"], "sqft": ["median"],
           "price_per_sqft": ["median"], "bedrooms": ["median"]}
    for col in ["gross_rental_yield_pct", "net_rental_yield_pct", "affordability_index",
                "est_monthly_mortgage", "mortgage_pct_of_income", "market_heat_index",
                "livability_score", "investment_score", "safety_score", "avg_school_rating",
                "median_dom", "avg_sale_to_list", "inventory", "months_of_supply",
                "yoy_price_change_pct", "median_sale_price", "median_household_income",
                "census_median_rent", "price_to_rent_ratio"]:
        if col in df.columns: agg[col] = ["median"]

    summary = df.groupby(["city", "state"]).agg(agg).round(2)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.reset_index()

    # Clean column names
    renames = {"price_median": "median_price", "price_mean": "avg_price",
               "price_min": "min_price", "price_max": "max_price",
               "price_count": "total_listings", "sqft_median": "median_sqft",
               "price_per_sqft_median": "median_ppsf", "bedrooms_median": "median_beds"}
    summary = summary.rename(columns=renames)
    for col in summary.columns:
        if col.endswith("_median"):
            new = col.replace("_median", "")
            if new not in summary.columns: summary = summary.rename(columns={col: new})

    if "investment_score" in summary.columns:
        summary = summary.sort_values("investment_score", ascending=False)

    summary.to_csv(OUT / "market_summary.csv", index=False)
    log.info(f"Saved: market_summary.csv ({len(summary)} cities)")

    # ── Print results ─────────────────────────────────────────────────────
    log.info(f"\nMarket Rankings:")
    show = ["city", "total_listings", "median_price"]
    for c in ["gross_rental_yield_pct", "yoy_price_change_pct", "safety_score",
              "livability_score", "investment_score"]:
        if c in summary.columns: show.append(c)
    log.info(f"\n{summary[show].to_string(index=False)}")

    log.info("\n✅ Done! Run: streamlit run app.py")


if __name__ == "__main__":
    main()