"""
analyze_reviews.py
- Loads reviews_clean.csv
- Runs sentiment (HuggingFace DistilBERT) with fallback to VADER
- Extracts TF-IDF keywords per bank and proposes theme buckets
- Saves processed CSV
"""
import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords
STOP = set(stopwords.words('english'))

INPUT_CSV = "data/reviews_clean.csv"
OUT_CSV = "data/reviews_processed.csv"

df = pd.read_csv(INPUT_CSV)
texts = df["review_text"].astype(str).tolist()

# ---------- Sentiment: try Hugging Face transformer pipeline ----------
use_hf = True
try:
    hf_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
except Exception as e:
    print("HuggingFace pipeline failed, falling back to VADER:", e)
    use_hf = False

vader = SentimentIntensityAnalyzer()

sent_labels = []
sent_scores = []

for text in tqdm(texts, desc="Sentiment"):
    if use_hf:
        try:
            out = hf_classifier(text[:512])[0]  # limit length
            label = out["label"].lower()
            score = float(out["score"])
        except Exception:
            # fallback for this text
            vs = vader.polarity_scores(text)
            if vs["compound"] >= 0.05:
                label, score = "positive", vs["compound"]
            elif vs["compound"] <= -0.05:
                label, score = "negative", -vs["compound"]
            else:
                label, score = "neutral", abs(vs["compound"])
    else:
        vs = vader.polarity_scores(text)
        if vs["compound"] >= 0.05:
            label, score = "positive", vs["compound"]
        elif vs["compound"] <= -0.05:
            label, score = "negative", -vs["compound"]
        else:
            label, score = "neutral", abs(vs["compound"])

    sent_labels.append(label)
    sent_scores.append(score)

df["sentiment_label"] = sent_labels
df["sentiment_score"] = sent_scores

# ---------- Thematic analysis: TF-IDF keywords per bank ----------
def preprocess(text):
    tokens = [w for w in text.lower().split() if w.isalpha() and w not in STOP]
    return " ".join(tokens)

df["clean_text"] = df["review_text"].astype(str).map(preprocess)

themes = {}
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
X = tfidf.fit_transform(df["clean_text"].fillna(""))

for bank in df["bank"].unique():
    idx = df["bank"] == bank
    Xb = X[idx.to_numpy()]
    sums = Xb.sum(axis=0)
    scores = [(word, sums[0, i]) for i, word in enumerate(tfidf.get_feature_names_out())]
    top = sorted(scores, key=lambda x: -x[1])[:40]
    themes[bank] = [w for w, s in top]

# Optional: run LDA for a quick topic proposal (3 topics per bank)
lda_topics = {}
for bank in df["bank"].unique():
    idx = df["bank"] == bank
    if idx.sum() < 50:
        lda_topics[bank] = []
        continue
    Xb = tfidf.transform(df.loc[idx, "clean_text"])
    lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=10)
    lda.fit(Xb)
    feature_names = tfidf.get_feature_names_out()
    bank_topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topn = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        bank_topics.append(topn)
    lda_topics[bank] = bank_topics

# Save outputs
df["proposed_themes"] = ""  # fill later with manual grouping
df.to_csv(OUT_CSV, index=False)

# Save theme suggestions for review (JSON or simple txt)
import json
with open("data/themes_suggestions.json","w") as f:
    json.dump({"tfidf_top_keywords": themes, "lda_topics": lda_topics}, f, indent=2)

print("[+] processed saved to", OUT_CSV)
print("[+] themes saved to data/themes_suggestions.json")
