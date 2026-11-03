# Limitations Statement
1. **Rating-to-label assumption:** a 3-star review isn't always "neutral"; some 5-star reviews contain complaints.
2. **Sarcasm & irony:** a bag-of-words model cannot understand them.
3. **Negation & mixed opinions:** "not bad", "good but..." are error-prone (see `results/error_rate_by_group.csv`).
4. **Domain shift:** a model trained on food reviews may fail on electronics or other writing styles.
5. **Language:** English only; slang, emojis and code-mixed text (e.g. Hinglish) are handled poorly.
6. **Dataset bias:** reviews skew positive; fake or incentivised reviews may exist.
7. **Use:** the tool gives estimates and must not be used for decisions about individual people or sellers without human review.
