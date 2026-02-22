import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models.sentiment import bert_emotion
from models.sentiment import cardiff
from models.sentiment import hartmann


def majority_vote(predictions):
    labels = [p for p in predictions if p in ("Positive", "Neutral", "Negative")]
    if not labels:
        return "UNKNOWN"
    return max(set(labels), key=labels.count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    input_path = output_dir / "preprocessed" / "processed_posts.json"
    output_path = output_dir / "preprocessed" / "sentiment_posts.csv"

    if not input_path.exists():
        print(f"No processed data found at {input_path}")
        print("Run process_reddit_posts.py first.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    labeled_posts = []

    for post in tqdm(posts, desc="Running sentiment analysis"):
        if not post.get("is_about_study", False):
            continue

        text = str(post.get("translated_text", "")).strip()
        if not text:
            continue

        cardiff_result = cardiff.classify(text)
        hartmann_result = hartmann.classify(text)
        bert_result = bert_emotion.classify(text)

        post["sentiment_cardiff"] = cardiff_result
        post["sentiment_hartmann"] = hartmann_result
        post["sentiment_bert_emotion"] = bert_result
        post["sentiment_majority"] = majority_vote([
            cardiff_result,
            hartmann_result,
            bert_result
        ])

        labeled_posts.append(post)

    df = pd.DataFrame(labeled_posts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(labeled_posts)} sentiment-labeled posts to '{output_path}'")


if __name__ == "__main__":
    main()