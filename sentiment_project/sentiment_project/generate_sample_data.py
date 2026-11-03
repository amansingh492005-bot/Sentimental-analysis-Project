"""Creates a SYNTHETIC review dataset (data/sample_reviews.csv) so the project runs offline.
Replace it with a real public dataset (see README) for your final submission."""
import random
from pathlib import Path

import pandas as pd

random.seed(42)
PRODUCTS = ["phone", "laptop", "blender", "headphones", "water bottle", "backpack", "keyboard",
            "coffee maker", "desk lamp", "vacuum cleaner", "camera", "smart watch"]
POS = ["fantastic", "excellent", "amazing", "sturdy", "reliable", "smooth", "great value", "impressive", "lovely"]
NEG = ["terrible", "awful", "flimsy", "useless", "disappointing", "overpriced", "unreliable", "poor quality", "noisy"]
ISSUES = ["the battery drains quickly", "the delivery was slow", "the packaging was damaged",
          "the instructions are confusing", "it feels a bit cheap"]
GOODS = ["the design is lovely", "setup was easy", "it looks great", "the price is fair"]

POS_T = ["I love this {p}. It is {a} and {b}.", "Absolutely {a}! The {p} works perfectly.",
         "Great {p}, would buy again. {a}.", "Not bad at all, this {p} is {a}.",
         "I can't complain, the {p} is {a}.", "Best {p} I have owned. {a} and {b}.",
         "Very happy with the {p}. Never had a problem."]
NEG_T = ["Terrible {p}. It is {a} and {b}.", "The {p} stopped working after a week. {a}.",
         "Not good. The {p} is {a}.", "I would not recommend this {p}. {a}.",
         "Never buying this {p} again, so {a}.", "Waste of money. {a} and {b}.",
         "The {p} is not worth it. Really {a}."]
NEU_T = ["The {p} is okay. Nothing special.", "It does the job, nothing more.",
         "Average {p}, some good points and some bad.", "The {p} is fine for the price, I guess.",
         "It is an ok {p}. Neither great nor terrible."]
MIX_T = ["The {p} is {a} but {i}.", "{g}, however {i}.", "Although {i}, the {p} is {a}."]
SARCASM = ["Oh great, another {p} that broke in two days.", "Yeah right, 'lasts a lifetime'. It died in a week."]
SHORT_P, SHORT_N, SHORT_U = ["Great!", "Love it", "Perfect."], ["Bad.", "Awful", "Do not buy"], ["Meh.", "It is ok", "Average"]


def make_review():
    r = random.choices([1, 2, 3, 4, 5], weights=[0.14, 0.10, 0.16, 0.22, 0.38])[0]
    p = random.choice(PRODUCTS)
    kind = random.random()
    if r >= 4:
        if kind < 0.08: txt = random.choice(SHORT_P)
        elif kind < 0.25 and r == 4: txt = random.choice(MIX_T).format(p=p, a=random.choice(POS), i=random.choice(ISSUES), g=random.choice(GOODS))
        else: txt = random.choice(POS_T).format(p=p, a=random.choice(POS), b=random.choice(POS))
    elif r <= 2:
        if kind < 0.08: txt = random.choice(SHORT_N)
        elif kind < 0.14: txt = random.choice(SARCASM).format(p=p)
        elif kind < 0.30 and r == 2: txt = random.choice(MIX_T).format(p=p, a=random.choice(NEG), i=random.choice(ISSUES), g=random.choice(GOODS))
        else: txt = random.choice(NEG_T).format(p=p, a=random.choice(NEG), b=random.choice(NEG))
    else:
        if kind < 0.15: txt = random.choice(SHORT_U)
        elif kind < 0.5: txt = random.choice(MIX_T).format(p=p, a=random.choice(POS + NEG), i=random.choice(ISSUES), g=random.choice(GOODS))
        else: txt = random.choice(NEU_T).format(p=p)
    if random.random() < 0.04:                       # label noise
        r = min(5, max(1, r + random.choice([-1, 1])))
    return {"review_text": txt, "rating": r}


if __name__ == "__main__":
    n = 4000
    df = pd.DataFrame([make_review() for _ in range(n)])
    out = Path("data/sample_reviews.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} synthetic reviews to {out}")
