"""Shared helpers: label mapping, text cleaning, data loading, explanations, error flags."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]

TEXT_CANDIDATES = ["review_text", "reviewtext", "text", "review", "content", "body", "comment"]
RATING_CANDIDATES = ["rating", "overall", "score", "stars", "star_rating", "star"]

NEGATIONS = {"not", "no", "never", "n't", "cannot", "cant", "can't", "won't", "wont",
             "didn't", "didnt", "isn't", "isnt", "don't", "dont", "doesn't", "doesnt",
             "wasn't", "wasnt", "hardly", "barely", "nothing"}
CONTRAST = {"but", "however", "although", "though", "yet", "except"}
SARCASM_HINTS = ["yeah right", "oh great", "oh sure", "as if", "thanks a lot", "just great",
                 "what a joke", "wow, so", "sure, because"]


def rating_to_sentiment(r):
    """Transparent mapping: 1-2 stars = negative, 3 = neutral, 4-5 = positive."""
    r = float(r)
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def clean_text(text):
    """Light cleaning. Negation words are deliberately KEPT (no stop-word removal)."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)            # html tags
    text = re.sub(r"http\S+|www\.\S+", " url ", text)  # urls
    text = re.sub(r"n't\b", " not", text)           # didn't -> did not
    text = re.sub(r"[^a-z0-9'\s!?]", " ", text)     # punctuation (keep ! and ?)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_col(df, candidates):
    lowered = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_dataset(path, max_rows=None, seed=42):
    """Load a CSV with a text column and a 1-5 rating column (auto-detected)."""
    df = pd.read_csv(path)
    text_col = _find_col(df, TEXT_CANDIDATES)
    rating_col = _find_col(df, RATING_CANDIDATES)
    if text_col is None or rating_col is None:
        raise ValueError(
            f"Could not find text/rating columns in {path}. Columns found: {list(df.columns)}. "
            "Rename them to 'review_text' and 'rating'.")
    df = df[[text_col, rating_col]].rename(columns={text_col: "review_text", rating_col: "rating"})
    df = df.dropna(subset=["review_text", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df[df["rating"].between(1, 5)].reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    return df


def ascii_ratio(text):
    text = str(text)
    return sum(ch.isascii() for ch in text) / max(len(text), 1)


def explain_prediction(pipeline, text, top_k=5):
    """Top terms pushing the linear model toward its predicted class."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    X = vec.transform([text])
    pred = clf.predict(X)[0]
    idx = list(clf.classes_).index(pred)
    contrib = X.multiply(clf.coef_[idx]).toarray()[0]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(contrib)[::-1][:top_k]
    return [(names[i], float(contrib[i])) for i in order if contrib[i] > 0]


def error_flags(text):
    words = clean_text(text).split()
    raw = str(text).lower()
    return {
        "short_review": len(words) <= 5,
        "has_negation": any(w in NEGATIONS for w in words),
        "mixed_opinion": any(w in CONTRAST for w in words),
        "possible_sarcasm": any(h in raw for h in SARCASM_HINTS),
    }
