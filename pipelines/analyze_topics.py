# pipelines/analyze_topics.py
import os
import pandas as pd
from tqdm import tqdm
from models.qa import topic_classifier

# Paths
INPUT_PATH = "data/preprocessed/sentiment_posts.csv"
OUTPUT_PATH = "data/final/final_posts.csv"

def main():
    if not os.path.exists(INPUT_PATH):
        print("❌ No sentiment data found. Please run analyze_sentiment.py first.")
        return

    # Load sentiment CSV
    df = pd.read_csv(INPUT_PATH)
    print(f"🎯 Classifying topics for {len(df)} posts...")

    # Add columns
    degree_types = []
    main_aspects = []

    for text in tqdm(df["translated_text"], desc="🔍 Topic Classification"):
        text = str(text).strip()
        degree_types.append(topic_classifier.get_most_likely_degree(text))
        main_aspects.append(topic_classifier.get_main_aspect_mentioned(text))

    df["degree_type"] = degree_types
    df["main_aspect"] = main_aspects

    # Save final CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n✅ Saved topic-annotated data to '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()
