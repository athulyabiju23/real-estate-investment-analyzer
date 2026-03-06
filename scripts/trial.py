import pandas as pd

df = pd.read_csv(
    "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/city_market_tracker.tsv000.gz",
    sep="\t", compression="gzip", dtype=str, nrows=5
)
print("Columns:", list(df.columns))
print("\nFirst row:")
print(df.iloc[0].to_string())