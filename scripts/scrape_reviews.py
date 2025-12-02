"""
scrape_reviews.py
- Fetch reviews for a list of Play Store package IDs (three banks)
- Save raw json and cleaned CSV
"""
import os
import time
import csv
from datetime import datetime
from retrying import retry
import pandas as pd
from google_play_scraper import reviews, reviews_all, Sort  # package: google-play-scraper

# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = "data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "reviews_clean.csv")
TARGET_PER_BANK = 600   # aim above 400 to be safe
SLEEP_BETWEEN_CALLS = 1.0

BANK_APPS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",   # ← replace with actual package id
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp"
}
SOURCE = "Google Play"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------
def normalize_date(ts):
    # expected input: datetime or ISO string
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            dt = datetime.strptime(ts, "%Y-%m-%d")
    else:
        dt = ts
    return dt.date().isoformat()  # YYYY-MM-DD

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def fetch_chunk(app_package, lang="en", country="us", sort=Sort.NEWEST, count=200, continuation_token=None):
    # wrapper around google_play_scraper.reviews to fetch pages
    result, _ = reviews(
        app_package,
        lang=lang,
        country=country,
        sort=sort,
        count=count,
        continuation_token=continuation_token
    )
    return result

# -------------------------
# MAIN
# -------------------------
all_rows = []

for bank_name, pkg in BANK_APPS.items():
    print(f"[+] Fetching reviews for {bank_name} ({pkg})")
    fetched = []
    cont_token = None
    attempts = 0
    # Some versions provide reviews_all() — try it first but fall back to chunked fetch
    try:
        print("  trying reviews_all() (may be limited for large apps)")
        raw = reviews_all(pkg, lang="en", country="us")
        for r in raw:
            fetched.append(r)
    except Exception as e:
        print("  reviews_all() failed / limited:", e)
    # If reviews_all did not reach target, do paginated attempts
    while len(fetched) < TARGET_PER_BANK:
        try:
            chunk = fetch_chunk(pkg, count=200)
        except Exception as e:
            print("  fetch_chunk error:", e)
            break
        if not chunk:
            print("  no more reviews returned; stopping.")
            break
        # dedupe by reviewId
        new = [c for c in chunk if c.get("reviewId") and c["reviewId"] not in {r.get("reviewId") for r in fetched}]
        if not new:
            print("  no new reviews in chunk; possibly reached end or being rate-limited.")
            break
        fetched.extend(new)
        print(f"  collected {len(fetched)} reviews so far for {bank_name}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    # transform
    for r in fetched:
        review_id = r.get("reviewId") or f"{pkg}_{hash(r.get('content',''))}"
        text = r.get("content") or ""
        rating = r.get("score") or r.get("rating") or None
        date_raw = r.get("at") or r.get("date") or None
        date_norm = normalize_date(date_raw)
        all_rows.append({
            "review_id": review_id,
            "review_text": text.strip(),
            "rating": int(rating) if rating is not None else None,
            "review_date": date_norm,
            "bank": bank_name,
            "app_package": pkg,
            "source": SOURCE
        })

# Save to CSV and do final cleaning
df = pd.DataFrame(all_rows)
print(f"[+] Total raw reviews fetched: {len(df)}")
# remove empty text
df = df[df["review_text"].str.strip().astype(bool)].drop_duplicates(subset=["review_text", "app_package"])
# If review_date missing, set to today (or NaN) — keep consistent
df["review_date"] = df["review_date"].fillna(pd.Timestamp.today().date().isoformat())
df.to_csv(CLEAN_CSV, index=False)
print(f"[+] Clean CSV saved to {CLEAN_CSV}")
