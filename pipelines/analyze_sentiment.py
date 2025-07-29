# pipelines/analyze_sentiment.py
import os
import json
import pandas as pd
from tqdm import tqdm

# Import classifiers
from models.sentiment import cardiff
from models.sentiment import hartmann
from models.sentiment import bert_emotion

# File paths
INPUT_PATH = "data/preprocessed/processed_posts.json"
OUTPUT_PATH = "data/preprocessed/sentiment_posts.csv"

# Load posts
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    posts = json.load(f)

# Majority vote function
def majority_vote(predictions):
    labels = [p for p in predictions if p in ("Positive", "Neutral", "Negative")]
    if not labels:
        return "UNKNOWN"
    return max(set(labels), key=labels.count)

# Process and filter posts
labeled_posts = []
for post in tqdm(posts, desc="🔍 Running sentiment analysis"):
    if not post.get("is_about_study", False):
        continue

    text = post.get("translated_text", "").strip()
    if not text:
        continue

    # Run sentiment classification
    cardiff_result = cardiff.classify(text)
    hartmann_result = hartmann.classify(text)
    bert_result = bert_emotion.classify(text)

    # Save all individual model results
    post["sentiment_cardiff"] = cardiff_result
    post["sentiment_hartmann"] = hartmann_result
    post["sentiment_bert_emotion"] = bert_result

    # Compute majority vote
    post["sentiment_majority"] = majority_vote([
        cardiff_result,
        hartmann_result,
        bert_result
    ])

    labeled_posts.append(post)

# Save to CSV
df = pd.DataFrame(labeled_posts)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n✅ Saved {len(labeled_posts)} sentiment-labeled posts to '{OUTPUT_PATH}'")