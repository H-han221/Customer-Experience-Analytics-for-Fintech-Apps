import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Paths
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR / "data"  # data is inside scripts
REPORTS_DIR = SCRIPTS_DIR.parent / "reports"  # reports in parent folder
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
print("Reports folder:", REPORTS_DIR.resolve())

CSV_FILE = DATA_DIR / "reviews_processed.csv"
THEMES_FILE = DATA_DIR / "themes_suggestions.json"

# Load data
df = pd.read_csv(CSV_FILE)
with open(THEMES_FILE) as f:
    themes = json.load(f)

# Ensure bank column exists
if 'bank' not in df.columns:
    df['bank'] = df['bank_name']  # fallback

# --- 1. Rating distribution by bank ---
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='rating', hue='bank')
plt.title("Rating distribution by bank")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.legend(title="Bank")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "rating_distribution.png", dpi=200)
plt.close()

# --- 2. Sentiment distribution by bank ---
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='sentiment_label', hue='bank', order=['positive','neutral','negative'])
plt.title("Sentiment distribution by bank")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.legend(title="Bank")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "sentiment_distribution.png", dpi=200)
plt.close()

# --- 3. Time series of average sentiment ---
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df_ts = df.dropna(subset=['review_date']).copy()
df_ts['week'] = df_ts['review_date'].dt.to_period('W')
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df_ts['sentiment_numeric'] = df_ts['sentiment_label'].map(sentiment_map)

plt.figure(figsize=(10,5))
for bank, group in df_ts.groupby('bank'):
    trend = group.groupby('week')['sentiment_numeric'].mean()
    trend.index = trend.index.to_timestamp()
    plt.plot(trend.index, trend.values, marker='o', label=bank)
plt.title("Weekly average sentiment trend by bank")
plt.xlabel("Week")
plt.ylabel("Average sentiment (-1=neg,0=neu,1=pos)")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "sentiment_trend.png", dpi=200)
plt.close()

# --- 4. Top 10 keywords per theme ---
for bank, bank_themes in themes.items():
    for theme_name, keywords in bank_themes.items():
        if not keywords:
            continue
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        kws, counts = zip(*top_keywords)
        # sanitize names for filename
        safe_bank = bank.replace(" ", "_")
        safe_theme = theme_name.replace(" ", "_")
        plt.figure(figsize=(8,5))
        sns.barplot(x=list(counts), y=list(kws), palette='viridis')
        plt.title(f"Top 10 keywords for {theme_name} ({bank})")
        plt.xlabel("Count")
        plt.ylabel("Keyword")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"top_keywords_{safe_bank}_{safe_theme}.png", dpi=200)
        plt.close()

print("[+] All plots saved in 'reports/'")
