"""
insert_into_postgres.py
- Connects to local/remote PostgreSQL and inserts banks + reviews from processed CSV
- Uses psycopg2 (recommended)
"""
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

CSV = "data/reviews_processed.csv"

DB_CONF = {
    "host": "localhost",
    "port": 5432,
    "dbname": "bank_reviews",
    "user": "youruser",
    "password": "yourpassword"
}

df = pd.read_csv(CSV)
conn = psycopg2.connect(**DB_CONF)
cur = conn.cursor()

# Insert banks and get bank_id mapping
banks = df[["bank", "app_package"]].drop_duplicates().values.tolist()
for name, pkg in banks:
    cur.execute("""
        INSERT INTO banks (bank_name, app_package)
        VALUES (%s, %s)
        ON CONFLICT (bank_name) DO UPDATE SET app_package = EXCLUDED.app_package
        RETURNING bank_id;
    """, (name, pkg))
conn.commit()

# build bank_id map
cur.execute("SELECT bank_id, bank_name FROM banks;")
rows = cur.fetchall()
bank_map = {name: bid for bid, name in [(r[0], r[1]) for r in rows]}

# prepare review rows
records = []
for _, r in df.iterrows():
    bid = bank_map[r["bank"]]
    records.append((
        r["review_id"],
        bid,
        r["review_text"],
        int(r["rating"]) if not pd.isna(r["rating"]) else None,
        r["review_date"] if not pd.isna(r["review_date"]) else None,
        r.get("sentiment_label"),
        float(r.get("sentiment_score")) if not pd.isna(r.get("sentiment_score")) else None,
        r.get("source")
    ))

# bulk insert using execute_values
sql = """
INSERT INTO reviews
  (review_id, bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, source)
VALUES %s
ON CONFLICT (review_id) DO NOTHING;
"""
execute_values(cur, sql, records, page_size=500)
conn.commit()
cur.close()
conn.close()
print("[+] Inserted", len(records), "reviews (ON CONFLICT DO NOTHING)")
