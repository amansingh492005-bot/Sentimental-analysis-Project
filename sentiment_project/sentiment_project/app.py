"""Review analyser prototype.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils import LABELS, error_flags, explain_prediction

st.set_page_config(page_title="Review Sentiment Analyser", page_icon="💬")
st.title("💬 Product Review Sentiment Analyser")
st.caption("Educational prototype. Predictions are model estimates, not facts about the reviewer.")

model_path = Path("models/analyzer_model.joblib")
if not model_path.exists():
    st.error("Model not found. Run `python sentiment_analysis.py` first.")
    st.stop()
model = joblib.load(model_path)

examples = ["Not bad at all, the phone is great value.",
            "The laptop is fantastic but the battery drains quickly.",
            "It is okay. Nothing special.",
            "Oh great, another blender that broke in two days.",
            "Terrible. Would not recommend."]
choice = st.selectbox("Try an example (or type your own below)", [""] + examples)
text = st.text_area("Review text", value=choice, height=130)

if st.button("Analyse", type="primary") and text.strip():
    proba = model.predict_proba([text])[0]
    classes = list(model.named_steps["clf"].classes_)
    probs = pd.Series(proba, index=classes).reindex(LABELS)
    pred, conf = probs.idxmax(), probs.max()
    emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}[pred]
    st.subheader(f"{emoji} Predicted sentiment: {pred}  ({conf:.0%} confidence)")
    st.bar_chart(probs)

    terms = explain_prediction(model, text)
    if terms:
        st.markdown("**Words/phrases that pushed the model toward this label:** " +
                    ", ".join(f"`{t}`" for t, _ in terms))
    else:
        st.markdown("_No known words/phrases influenced this prediction (text may be unfamiliar)._")

    notes = []
    if conf < 0.55:
        notes.append("Low confidence - the model is unsure; treat this result with extra caution.")
    f = error_flags(text)
    if f["short_review"]: notes.append("Very short reviews give the model little to work with.")
    if f["has_negation"]: notes.append("Negation (e.g. 'not good') is a known weak spot.")
    if f["mixed_opinion"]: notes.append("Mixed opinions ('good but...') are often labelled neutral or misread.")
    if f["possible_sarcasm"]: notes.append("Possible sarcasm detected - the model cannot reliably understand sarcasm.")
    st.info("⚠️ **Caution:** automated opinion analysis can be wrong. It was trained on star-rating-derived "
            "labels and may not transfer to other product types or writing styles.")
    for n in notes:
        st.warning(n)

meta = Path("models/meta.json")
if meta.exists():
    with st.expander("About the model"):
        st.json(json.loads(meta.read_text()))
