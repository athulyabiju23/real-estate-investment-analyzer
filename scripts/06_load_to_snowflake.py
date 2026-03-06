"""
Upload final CSVs to Snowflake.
"""

import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

# ── Connect ───────────────────────────────────────────────────────────────────

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    database="real_estate",
    warehouse="re_warehouse",
    schema="analytics",
)
cur = conn.cursor()
print("Connected to Snowflake")


# ── Upload function ───────────────────────────────────────────────────────────

def upload(csv_path, table_name):
    df = pd.read_csv(csv_path)
    print(f"\nLoading {csv_path}: {len(df)} rows")

    cur.execute(f"TRUNCATE TABLE IF EXISTS {table_name}")

    # Get table columns from Snowflake
    cur.execute(f"SHOW COLUMNS IN TABLE {table_name}")
    sf_cols = [row[2].lower() for row in cur.fetchall()]

    # Match CSV columns to Snowflake columns
    df.columns = df.columns.str.strip().str.lower()
    common = [c for c in sf_cols if c in df.columns]
    df = df[common]
    df = df.where(pd.notnull(df), None)

    # Insert
    cols = ", ".join(common)
    placeholders = ", ".join(["%s"] * len(common))
    sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

    data = [tuple(None if pd.isna(v) else v for v in row) for row in df.values]

    batch = 500
    for i in range(0, len(data), batch):
        cur.executemany(sql, data[i:i+batch])
        print(f"  Inserted {min(i+batch, len(data))}/{len(data)}")

    conn.commit()
    print(f"✓ {table_name}: {len(data)} rows loaded")


# ── Upload both files ─────────────────────────────────────────────────────────

upload("data/final/properties_enriched.csv", "properties")
upload("data/final/market_summary.csv", "market_summary")

# Verify
for table in ["properties", "market_summary"]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"\n{table}: {cur.fetchone()[0]} rows in Snowflake")

cur.close()
conn.close()
print("\n✅ Done! Data is in Snowflake.")