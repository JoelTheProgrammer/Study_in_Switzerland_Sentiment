import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from models.qa import topic_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    input_path = output_dir / "preprocessed" / "sentiment_posts.csv"
    output_path = output_dir / "final" / "final_posts.csv"

    if not input_path.exists():
        print(f"No sentiment data found at {input_path}")
        print("Run analyze_sentiment.py first.")
        return

    topic_classifier.load_topic_classifier_config(input_dir)

    df = pd.read_csv(input_path)
    print(f"Classifying topics for {len(df)} posts...")

    degree_types = []
    main_aspects = []

    for text in tqdm(df["translated_text"], desc="Topic Classification"):
        text = str(text).strip()
        degree_types.append(topic_classifier.get_most_likely_degree(text))
        main_aspects.append(topic_classifier.get_main_aspect(text))

    df["degree_type"] = degree_types
    df["main_aspect"] = main_aspects

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved topic-annotated data to '{output_path}'")


if __name__ == "__main__":
    main()