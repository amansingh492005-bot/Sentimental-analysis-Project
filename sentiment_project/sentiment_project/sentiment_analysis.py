"""Capstone 4 - Sentiment Analysis of Product Reviews (full pipeline).

Usage:
    python sentiment_analysis.py                          # uses data/reviews.csv, else data/sample_reviews.csv
    python sentiment_analysis.py --data data/my.csv --max-rows 50000
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import LABELS, ascii_ratio, clean_text, error_flags, load_dataset

SEED = 42
RES, MOD = Path("results"), Path("models")
RES.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)


def tfidf():
    # Negations kept (no stop-word removal); unigrams + bigrams catch "not good".
    return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                           max_df=0.95, sublinear_tf=True)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def inspect(df):
    info = {
        "rows": int(len(df)),
        "duplicate_texts": int(df["review_text"].duplicated().sum()),
        "label_counts": df["sentiment"].value_counts().to_dict(),
        "rating_counts": {int(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()},
        "median_words": float(df["review_text"].str.split().str.len().median()),
        "mean_words": float(df["review_text"].str.split().str.len().mean()),
        "pct_mostly_non_english": float((df["review_text"].apply(ascii_ratio) < 0.8).mean() * 100),
        "pct_very_short(<=3 words)": float((df["review_text"].str.split().str.len() <= 3).mean() * 100),
    }
    print(json.dumps(info, indent=2))
    (RES / "data_inspection.json").write_text(json.dumps(info, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df["sentiment"].value_counts().reindex(LABELS).plot.bar(ax=ax[0], color=["#d9534f", "#f0ad4e", "#5cb85c"])
    ax[0].set_title("Class distribution"); ax[0].set_ylabel("Reviews"); ax[0].tick_params(axis="x", rotation=0)
    df["review_text"].str.split().str.len().clip(upper=200).plot.hist(bins=40, ax=ax[1], color="#5b8def")
    ax[1].set_title("Review length (words, capped at 200)"); ax[1].set_xlabel("Words")
    plt.tight_layout(); plt.savefig(RES / "01_data_overview.png", dpi=150); plt.close()
    return info


def top_frequent_words(df, k=12):
    stop = set(ENGLISH_STOP_WORDS) - {"not", "no", "never"}
    out = {}
    for lab in LABELS:
        c = Counter(w for t in df.loc[df.sentiment == lab, "review_text"]
                    for w in clean_text(t).split() if w not in stop and len(w) > 2)
        out[lab] = c.most_common(k)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, lab, colr in zip(axes, LABELS, ["#d9534f", "#f0ad4e", "#5cb85c"]):
        words, counts = zip(*out[lab][::-1])
        ax.barh(words, counts, color=colr); ax.set_title(f"Most frequent words: {lab}")
    plt.tight_layout(); plt.savefig(RES / "02_frequent_words.png", dpi=150); plt.close()


def top_model_terms(lr_pipe, k=12):
    vec, clf = lr_pipe.named_steps["tfidf"], lr_pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    rows = []
    for ax, lab, colr in zip(axes, LABELS, ["#d9534f", "#f0ad4e", "#5cb85c"]):
        i = list(clf.classes_).index(lab)
        top = np.argsort(clf.coef_[i])[::-1][:k]
        ax.barh(names[top][::-1], clf.coef_[i][top][::-1], color=colr)
        ax.set_title(f"Top phrases for '{lab}'")
        rows += [(lab, names[j], float(clf.coef_[i][j])) for j in top]
    plt.tight_layout(); plt.savefig(RES / "03_top_phrases_per_class.png", dpi=150); plt.close()
    pd.DataFrame(rows, columns=["class", "phrase", "weight"]).to_csv(RES / "top_phrases.csv", index=False)


def main(args):
    data_path = Path(args.data) if args.data else (
        Path("data/reviews.csv") if Path("data/reviews.csv").exists() else Path("data/sample_reviews.csv"))
    if not data_path.exists():
        import generate_sample_data  # noqa: F401
        raise SystemExit("Run: python generate_sample_data.py  (or place your CSV at data/reviews.csv)")
    print(f"\n=== 1. Loading {data_path} ===")
    df = load_dataset(data_path, max_rows=args.max_rows, seed=SEED)

    print("\n=== 2. Data inspection ===")
    inspect(df)
    df = df.drop_duplicates(subset="review_text").reset_index(drop=True)
    print(f"After removing duplicate texts: {len(df)} rows")
    top_frequent_words(df)

    print("\n=== 3. Stratified train/test split (split BEFORE fitting TF-IDF) ===")
    X_tr, X_te, y_tr, y_te = train_test_split(df["review_text"], df["sentiment"], test_size=0.2,
                                              stratify=df["sentiment"], random_state=SEED)
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    print("\n=== 4. Training baseline + models ===")
    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes (Complement)": ComplementNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0),
        "Linear SVM": LinearSVC(class_weight="balanced", C=0.5),
    }
    fitted, rows, per_class = {}, [], []
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rep = classification_report(y_te, pred, labels=LABELS, output_dict=True, zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(y_te, pred),
                     "macro_f1": f1_score(y_te, pred, average="macro"),
                     "weighted_f1": f1_score(y_te, pred, average="weighted")})
        for lab in LABELS:
            per_class.append({"model": name, "class": lab, "precision": rep[lab]["precision"],
                              "recall": rep[lab]["recall"], "f1": rep[lab]["f1-score"],
                              "support": int(rep[lab]["support"])})
        fitted[name] = pipe
        print(f"{name:28s} acc={rows[-1]['accuracy']:.3f}  macro-F1={rows[-1]['macro_f1']:.3f}")

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    classwise = pd.DataFrame(per_class)
    summary.to_csv(RES / "model_comparison.csv", index=False)
    classwise.to_csv(RES / "classwise_metrics.csv", index=False)
    best_name = summary.iloc[0]["model"]
    print(f"\nBest model by macro-F1: {best_name}")

    # Model comparison chart
    ax = summary.set_index("model")[["accuracy", "macro_f1"]].plot.bar(figsize=(8, 4.5), rot=15)
    ax.set_title("Model comparison (test set)"); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(RES / "04_model_comparison.png", dpi=150); plt.close()

    # Class-wise chart + confusion matrices
    best_pipe = fitted[best_name]
    best_pred = best_pipe.predict(X_te)
    cw = classwise[classwise.model == best_name].set_index("class")[["precision", "recall", "f1"]].reindex(LABELS)
    cw.plot.bar(figsize=(7, 4), rot=0, title=f"Per-class metrics - {best_name}"); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(RES / "05_classwise_metrics.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, best_pred, labels=LABELS), display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name}")
    plt.tight_layout(); plt.savefig(RES / "06_confusion_matrix.png", dpi=150); plt.close()
    print("\nClassification report (best model):\n",
          classification_report(y_te, best_pred, labels=LABELS, zero_division=0))

    print("=== 5. Word/phrase visualisation ===")
    top_model_terms(fitted["Logistic Regression"])

    print("=== 6. Error analysis (negation, sarcasm, mixed opinions, short reviews) ===")
    err = pd.DataFrame({"review_text": X_te.values, "true": y_te.values, "predicted": best_pred})
    flags = pd.DataFrame([error_flags(t) for t in err.review_text])
    err = pd.concat([err, flags], axis=1)
    err["is_error"] = err["true"] != err["predicted"]
    err[err.is_error].to_csv(RES / "misclassified_examples.csv", index=False)
    fl_rows = [{"group": "ALL reviews", "n": len(err), "error_rate": err.is_error.mean()}]
    for f in flags.columns:
        sub = err[err[f]]
        if len(sub):
            fl_rows.append({"group": f, "n": len(sub), "error_rate": sub.is_error.mean()})
    flag_df = pd.DataFrame(fl_rows)
    flag_df.to_csv(RES / "error_rate_by_group.csv", index=False)
    print(flag_df.to_string(index=False))

    print("\n=== 7. Saving models ===")
    joblib.dump(fitted["Logistic Regression"], MOD / "analyzer_model.joblib")  # explainable + probabilities
    joblib.dump(best_pipe, MOD / "best_model.joblib")
    (MOD / "meta.json").write_text(json.dumps({"best_model": best_name, "labels": LABELS,
                                                "data_file": str(data_path)}, indent=2))

    report = f"""# Evaluation Report (auto-generated)

Data file: `{data_path}` | Rows used (after de-duplication): {len(df)} | Test size: {len(X_te)}

## Model comparison (test set)
{md_table(summary)}

## Per-class metrics
{md_table(classwise)}

## Error rate by review type ({best_name})
{md_table(flag_df)}

**Why macro-F1?** Neutral reviews are usually the minority and hardest class. Accuracy alone hides poor
neutral performance, so macro-F1 (equal weight per class) and per-class precision/recall are reported.

Figures: see `results/*.png`. Misclassified examples: `results/misclassified_examples.csv`.
"""
    (RES / "evaluation_report.md").write_text(report, encoding="utf-8")
    print("\nDone. Outputs are in results/ and models/. Launch the app with: streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV with review text + 1-5 rating columns")
    ap.add_argument("--max-rows", type=int, default=50000, help="Cap rows for speed (0 = no cap)")
    a = ap.parse_args()
    a.max_rows = a.max_rows or None
    main(a)
