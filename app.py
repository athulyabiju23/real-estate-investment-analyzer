"""
Real Estate Investment Dashboard
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import snowflake.connector

st.set_page_config(page_title="RE Investment Analyzer", page_icon="🏠", layout="wide")

# ─── Load Data from Snowflake ─────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    conn = snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        database="real_estate",
        warehouse="re_warehouse",
        schema="analytics",
    )
    props = pd.read_sql("SELECT * FROM properties", conn)
    market = pd.read_sql("SELECT * FROM market_summary", conn)
    conn.close()
    # Snowflake returns UPPERCASE column names
    props.columns = props.columns.str.lower()
    market.columns = market.columns.str.lower()
    return props, market

df, market = load_data()
COLORS = px.colors.qualitative.Set2

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🏠 Filters")
all_cities = sorted(df["city"].unique())
st.sidebar.markdown("**Select Cities:**")
select_all = st.sidebar.checkbox("All Cities", value=True)
cities = []
if select_all:
    cities = all_cities
else:
    for c in all_cities:
        if st.sidebar.checkbox(c, value=False):
            cities.append(c)
if not cities:
    cities = all_cities
price_range = st.sidebar.slider("Price ($)", int(df["price"].min()), int(df["price"].max()),
                                 (int(df["price"].min()), int(df["price"].max())), step=10000, format="$%d")
beds = st.sidebar.slider("Bedrooms", int(df["bedrooms"].min()), int(df["bedrooms"].max()),
                          (int(df["bedrooms"].min()), int(df["bedrooms"].max())))

f = df[(df["city"].isin(cities)) & (df["price"].between(*price_range)) & (df["bedrooms"].between(*beds))]
fm = market[market["city"].isin(cities)]
st.sidebar.markdown(f"**{len(f):,} properties | {f['city'].nunique()} markets**")

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🏠 U.S. Real Estate Investment Analyzer")
st.caption("Data: Redfin, Realtor, Census Bureau, FBI UCR, GreatSchools")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median Price", f"${f['price'].median():,.0f}")
c2.metric("Median $/Sqft", f"${f['price_per_sqft'].median():,.0f}")
if "gross_rental_yield_pct" in f.columns:
    c3.metric("Median Yield", f"{f['gross_rental_yield_pct'].median():.1f}%")
if "investment_score" in f.columns:
    c4.metric("Median Inv Score", f"{f['investment_score'].median():.1f}")

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💰 Investment", "🏡 Livability", "🏘️ Compare", "🔍 Table"])
T = "plotly_dark"

# ═══ OVERVIEW ═════════════════════════════════════════════════════════════════

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(f, x="city", y="price", color="city", template=T, title="Price Distribution")
        fig.update_layout(showlegend=False, height=400)
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "yoy_price_change_pct" in fm.columns:
            s = fm.sort_values("yoy_price_change_pct")
            colors = ["#ef4444" if x < 0 else "#22c55e" for x in s["yoy_price_change_pct"]]
            fig = go.Figure(go.Bar(x=s["yoy_price_change_pct"], y=s["city"], orientation="h",
                                    marker_color=colors,
                                    text=s["yoy_price_change_pct"].apply(lambda x: f"{x:+.1f}%"),
                                    textposition="outside"))
            fig.update_layout(title="YoY Price Change", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if "median_dom" in fm.columns:
            s = fm.sort_values("median_dom")
            fig = go.Figure(go.Bar(x=s["median_dom"], y=s["city"], orientation="h",
                                    marker_color=px.colors.sequential.Oranges_r[:len(s)],
                                    text=s["median_dom"].apply(lambda x: f"{x:.0f} days"),
                                    textposition="outside"))
            fig.update_layout(title="Days on Market", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "months_of_supply" in fm.columns:
            s = fm.sort_values("months_of_supply")
            fig = go.Figure(go.Bar(x=s["months_of_supply"], y=s["city"], orientation="h",
                                    marker_color=px.colors.sequential.Blues[:len(s)],
                                    text=s["months_of_supply"].apply(lambda x: f"{x:.1f} mo"),
                                    textposition="outside"))
            fig.add_vline(x=4, line_dash="dash", line_color="gray", annotation_text="Balanced")
            fig.update_layout(title="Months of Supply", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

# ═══ INVESTMENT ═══════════════════════════════════════════════════════════════

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        if "gross_rental_yield_pct" in f.columns:
            fig = px.scatter(f, x="price", y="gross_rental_yield_pct", color="city", size="sqft",
                              title="Price vs Rental Yield", template=T, hover_data=["bedrooms", "bathrooms"])
            fig.update_xaxes(tickformat="$,.0f")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "investment_score" in f.columns:
            fig = px.scatter(f, x="price", y="investment_score", color="city", size="sqft",
                              title="Price vs Investment Score", template=T)
            fig.update_xaxes(tickformat="$,.0f")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if "gross_rental_yield_pct" in fm.columns:
            s = fm.sort_values("gross_rental_yield_pct")
            fig = go.Figure(go.Bar(x=s["gross_rental_yield_pct"], y=s["city"], orientation="h",
                                    marker_color=px.colors.sequential.Greens[:len(s)],
                                    text=s["gross_rental_yield_pct"].apply(lambda x: f"{x:.1f}%"),
                                    textposition="outside"))
            fig.update_layout(title="Rental Yield by City", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "mortgage_pct_of_income" in fm.columns:
            s = fm.sort_values("mortgage_pct_of_income")
            colors = ["#ef4444" if x > 30 else "#22c55e" for x in s["mortgage_pct_of_income"]]
            fig = go.Figure(go.Bar(x=s["mortgage_pct_of_income"], y=s["city"], orientation="h",
                                    marker_color=colors,
                                    text=s["mortgage_pct_of_income"].apply(lambda x: f"{x:.0f}%"),
                                    textposition="outside"))
            fig.add_vline(x=30, line_dash="dash", line_color="#ef4444", annotation_text="30% limit")
            fig.update_layout(title="Mortgage as % of Income", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Top picks
    st.subheader("🎯 Top Picks")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**💰 Cash Flow**")
        if "gross_rental_yield_pct" in fm.columns:
            for _, r in fm.nlargest(3, "gross_rental_yield_pct").iterrows():
                st.write(f"**{r['city']}** — {r['gross_rental_yield_pct']:.1f}% yield, ${r.get('median_price', 0):,.0f}")
    with c2:
        st.markdown("**📈 Growth**")
        if "yoy_price_change_pct" in fm.columns:
            for _, r in fm.nlargest(3, "yoy_price_change_pct").iterrows():
                st.write(f"**{r['city']}** — {r['yoy_price_change_pct']:+.1f}% YoY")
    with c3:
        st.markdown("**🏷️ Budget (<$400K)**")
        if "median_price" in fm.columns and "investment_score" in fm.columns:
            for _, r in fm[fm["median_price"] <= 400000].nlargest(3, "investment_score").iterrows():
                st.write(f"**{r['city']}** — score {r['investment_score']:.0f}, ${r['median_price']:,.0f}")

# ═══ LIVABILITY ═══════════════════════════════════════════════════════════════

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        if "safety_score" in fm.columns:
            s = fm.sort_values("safety_score")
            fig = go.Figure(go.Bar(x=s["safety_score"], y=s["city"], orientation="h",
                                    marker_color=px.colors.sequential.RdBu[3:3+len(s)],
                                    text=s["safety_score"].apply(lambda x: f"{x:.0f}"),
                                    textposition="outside"))
            fig.update_layout(title="Safety Score (higher = safer)", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "avg_school_rating" in fm.columns:
            s = fm.sort_values("avg_school_rating")
            fig = go.Figure(go.Bar(x=s["avg_school_rating"], y=s["city"], orientation="h",
                                    marker_color=px.colors.sequential.Purples[:len(s)],
                                    text=s["avg_school_rating"].apply(lambda x: f"{x:.1f}/10"),
                                    textposition="outside"))
            fig.update_layout(title="School Rating", template=T, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Radar
    if "livability_score" in fm.columns:
        st.subheader("City Profiles")
        top5 = fm.nlargest(5, "livability_score")
        metrics = {"safety_score": ("Safety", 100), "avg_school_rating": ("Schools", 10),
                   "livability_score": ("Livability", 100)}
        if "market_heat_index" in top5.columns:
            metrics["market_heat_index"] = ("Heat", 100)
        if "gross_rental_yield_pct" in top5.columns:
            metrics["gross_rental_yield_pct"] = ("Yield", 15)

        fig = go.Figure()
        for _, row in top5.iterrows():
            vals, labels = [], []
            for col, (lbl, mx) in metrics.items():
                if col in row.index and pd.notna(row[col]):
                    vals.append(row[col] / mx * 100)
                    labels.append(lbl)
            vals.append(vals[0])
            labels.append(labels[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=labels, name=row["city"], fill="toself", opacity=0.5))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          template=T, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ═══ COMPARE PROPERTIES ══════════════════════════════════════════════════════

with tab4:
    st.subheader("Compare Properties Within a City")
    sel_city = st.selectbox("Select City", sorted(f["city"].unique()))
    city_data = f[f["city"] == sel_city]

    if len(city_data) == 0:
        st.warning("No properties for this city.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            bed_prices = city_data.groupby("bedrooms")["price"].median().reset_index()
            fig = go.Figure(go.Bar(x=bed_prices["bedrooms"], y=bed_prices["price"],
                                    marker_color=COLORS[:len(bed_prices)],
                                    text=bed_prices["price"].apply(lambda x: f"${x:,.0f}"),
                                    textposition="outside"))
            fig.update_layout(title=f"Median Price by Bedrooms — {sel_city}", template=T, height=400,
                              yaxis_tickformat="$,.0f", xaxis_title="Bedrooms")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(city_data, x="sqft", y="price", color="bedrooms", size="bathrooms",
                              title=f"Sqft vs Price — {sel_city}", template=T,
                              hover_data=["full_address"], color_continuous_scale="Viridis")
            fig.update_yaxes(tickformat="$,.0f")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(city_data, x="price", nbins=20, title=f"Price Distribution — {sel_city}",
                                template=T, color_discrete_sequence=["#3b82f6"])
            fig.update_xaxes(tickformat="$,.0f")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(city_data, x="price_per_sqft", nbins=20,
                                title=f"$/Sqft Distribution — {sel_city}",
                                template=T, color_discrete_sequence=["#8b5cf6"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Properties", len(city_data))
        c2.metric("Median Price", f"${city_data['price'].median():,.0f}")
        c3.metric("Price Range", f"${city_data['price'].min():,.0f}–${city_data['price'].max():,.0f}")
        c4.metric("Median $/Sqft", f"${city_data['price_per_sqft'].median():,.0f}")

# ═══ TABLE ════════════════════════════════════════════════════════════════════

with tab5:
    sort_by = st.selectbox("Sort by", ["investment_score", "price", "gross_rental_yield_pct", "sqft"])
    asc = st.checkbox("Ascending")

    cols = ["city", "price", "bedrooms", "bathrooms", "sqft", "price_per_sqft", "listing_status"]
    for c in ["gross_rental_yield_pct", "investment_score", "investment_tier", "pct_vs_market"]:
        if c in f.columns:
            cols.append(c)

    display = f[cols].sort_values(sort_by, ascending=asc) if sort_by in f.columns else f[cols]
    st.dataframe(display, use_container_width=True, height=600)
    st.download_button("📥 Download CSV", f.to_csv(index=False), "properties.csv", "text/csv")

# ─── Correlation ──────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📐 Correlations")
corr_cols = [c for c in ["price", "gross_rental_yield_pct", "investment_score",
                          "market_heat_index", "livability_score", "safety_score"] if c in f.columns]
if len(corr_cols) >= 3:
    fig = px.imshow(f[corr_cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", template=T)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Sources: Redfin, Realtor.com, U.S. Census Bureau, FBI UCR, GreatSchools")