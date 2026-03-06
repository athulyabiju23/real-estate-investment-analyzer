# 🏠 U.S. Real Estate Market Intelligence Dashboard

**A data-driven analysis of the fastest-growing U.S. real estate markets in 2025**

Built as a portfolio project demonstrating end-to-end data analytics — from web scraping and data engineering to metric calculation and interactive visualization.

---

## Project Overview

This project identifies high-growth real estate markets across the U.S. by collecting property data from Zillow and Redfin, engineering key investment metrics, and presenting actionable insights through an interactive dashboard.

### Key Metrics Calculated
- **Annual Growth Rate** — YoY property price appreciation per market
- **Rental Yield** — Annual rent / property price, measuring cash flow potential
- **Affordability Index** — Median home price / median household income
- **Safety Index** — Composite score from crime statistics
- **Livability Score** — Weighted composite of schools, amenities, safety, and transit

### Target Audience
- **Cash Flow Investors** → Markets with highest rental yields
- **Growth Investors** → Markets with strongest price appreciation
- **Budget-Conscious Buyers** → Affordable markets with upside potential

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Collection | Python, BeautifulSoup, Selenium, Zillow API (via RapidAPI) |
| Data Processing | Pandas, NumPy |
| Analysis | Scipy, Scikit-learn |
| Visualization | Plotly, Streamlit (interactive dashboard) |
| Portfolio Demo | React + Recharts (standalone dashboard) |

---

## Project Structure

```
real-estate-project/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_data_collection.py      # Scrape Zillow/Redfin data
│   ├── 02_data_cleaning.py        # Clean & standardize raw data
│   ├── 03_feature_engineering.py  # Calculate all investment metrics
│   ├── 04_eda_analysis.py         # Exploratory data analysis
│   └── 05_generate_demo_data.py   # Generate realistic demo dataset
├── data/
│   ├── raw/                       # Raw scraped data (gitignored)
│   ├── cleaned/                   # Cleaned datasets
│   └── final/                     # Analysis-ready datasets
├── notebooks/
│   └── exploration.ipynb          # Jupyter notebook for EDA
└── dashboard/
    └── dashboard.jsx              # Interactive React dashboard
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Demo Data (no API keys needed)
```bash
python scripts/05_generate_demo_data.py
```

### 3. Run Full Pipeline with Real Data
```bash
# Set your RapidAPI key for Zillow API
export RAPIDAPI_KEY="your_key_here"

python scripts/01_data_collection.py
python scripts/02_data_cleaning.py
python scripts/03_feature_engineering.py
python scripts/04_eda_analysis.py
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

---

## Key Findings (Demo Data)

1. **Highest Rental Yields**: Jacksonville, FL and Tampa, FL lead with 6%+ yields
2. **Fastest Appreciation**: Austin, TX and Boise, ID show 12%+ annual growth
3. **Best Value Markets**: Raleigh, NC and Charlotte, NC balance affordability with growth
4. **Hidden Gems**: Salt Lake City and Nashville offer strong fundamentals at moderate prices

---

## License
MIT — Built for educational and portfolio purposes.
