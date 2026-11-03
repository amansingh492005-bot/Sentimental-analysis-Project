# Capstone 4 — Sentiment Analysis of Product Reviews

Classifies product reviews as **negative / neutral / positive** using TF-IDF + classical ML, with a baseline,
model comparison, per-class evaluation, error analysis and a Streamlit review analyser.

## Project structure
```
sentiment_project/
├── data/                  # put your real dataset here as reviews.csv
├── docs/                  # dataset card, prompt log, limitations (fill these in)
├── results/               # generated charts, metrics, report
├── models/                # generated saved models
├── utils.py               # label mapping, text cleaning, loader, explanations
├── generate_sample_data.py# creates SYNTHETIC demo data (offline fallback)
├── sentiment_analysis.py  # full pipeline (run this)
├── app.py                 # Streamlit review analyser
├── test_robustness.py     # example test cases
└── requirements.txt
```

## Setup in VS Code
1. Install **Python 3.9+** and the VS Code **Python** extension.
2. `File > Open Folder…` → select `sentiment_project`.
3. Open the terminal (`Ctrl+` `` ` ``) and create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:  venv\Scripts\activate
   # Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
4. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `venv` one.

## Run
```bash
python generate_sample_data.py      # optional: synthetic demo data (skip if you have a real dataset)
python sentiment_analysis.py        # trains models, writes results/ and models/
python test_robustness.py           # example test cases
streamlit run app.py                # opens the analyser in your browser
```

## Use a REAL public dataset (needed for your final submission)
The included sample data is **synthetic** — results on it are far too clean. Download one of these, save as `data/reviews.csv`:
- **Amazon Fine Food Reviews** (Kaggle) — columns `Text`, `Score`
- **Amazon Cell Phones & Accessories reviews** (Kaggle) — star-rated, 1–5
- **Yelp / Flipkart / Amazon Cell-Phone reviews** from Kaggle

The loader auto-detects text columns (`review_text`, `Text`, `review`…) and rating columns (`rating`, `Score`, `overall`, `stars`…).
For big files: `python sentiment_analysis.py --max-rows 50000` (default cap). Record source + licence in `docs/DATASET_CARD.md`.

## How this maps to the 12 capstone steps
| # | Requirement | Where |
|---|---|---|
| 1–2 | Public dataset, source & licence | `data/`, `docs/DATASET_CARD.md` |
| 3 | Label distribution, duplicates, language, length | `inspect()` → `results/data_inspection.json`, `01_data_overview.png` |
| 4 | Rating → sentiment mapping | `utils.rating_to_sentiment` (1–2 neg, 3 neutral, 4–5 pos) |
| 5 | Stratified split + consistent preprocessing | `sentiment_analysis.py` (split before TF-IDF; one `Pipeline`) |
| 6 | TF-IDF baseline + ≥2 algorithms | Majority baseline, Complement NB, Logistic Regression, Linear SVM |
| 7 | Per-class precision/recall/F1 | `results/classwise_metrics.csv`, `05_classwise_metrics.png`, `06_confusion_matrix.png` |
| 8 | Common words/phrases per class | `02_frequent_words.png`, `03_top_phrases_per_class.png` |
| 9 | Errors: sarcasm, mixed, short, negation | `results/misclassified_examples.csv`, `error_rate_by_group.csv` |
| 10 | Review analyser with cautious explanation | `app.py` |
| 11 | Assumptions, domain shift, limits | `docs/LIMITATIONS.md` |
| 12 | Prompts, experiments, reproducibility | `docs/PROMPT_LOG.md`, this README (fixed seed = 42) |

## Design decisions (good to mention in your viva)
- **No stop-word removal**: words like *not* flip meaning. Bigrams capture “not good”.
- **Split before vectorising** so TF-IDF never sees test data (no leakage).
- **Macro-F1** is the main metric because neutral is small and hardest; accuracy alone is misleading.
- **Class weights = balanced** to stop the model ignoring the neutral class.
- The app uses Logistic Regression because it gives probabilities and explainable word weights.

## Suggested improvements
Try a DistilBERT baseline in Google Colab, tune hyperparameters with cross-validation, test domain shift
(train on one product category, test on another), or add VADER as a lexicon baseline.

## Submission checklist
- [ ] Dataset card (`docs/DATASET_CARD.md`) · [ ] Notebook/script (`sentiment_analysis.py`) · [ ] Class-wise evaluation
- [ ] Word/phrase visualisations · [ ] Analyser prototype (`app.py`) · [ ] Report + presentation + prompt log
