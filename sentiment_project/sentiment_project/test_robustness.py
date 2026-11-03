"""Quick example test cases (negation, mixed, sarcasm, short, neutral). Run: python test_robustness.py"""
import joblib

from utils import LABELS

CASES = [
    ("This is great, I love it!", "positive"),
    ("Terrible product, complete waste of money.", "negative"),
    ("It is okay, nothing special.", "neutral"),
    ("Not good at all.", "negative"),
    ("Not bad, actually very good.", "positive"),
    ("Great phone but the battery is terrible.", "neutral"),     # mixed - ambiguous by design
    ("Oh great, another one that broke in two days.", "negative"),  # sarcasm
    ("Meh.", "neutral"),
    ("Love it", "positive"),
]

model = joblib.load("models/analyzer_model.joblib")
ok = 0
for text, expected in CASES:
    pred = model.predict([text])[0]
    ok += pred == expected
    print(f"{'PASS' if pred == expected else 'FAIL'} | expected={expected:8s} got={pred:8s} | {text}")
print(f"\n{ok}/{len(CASES)} passed (failures are informative - discuss them in your report).")
