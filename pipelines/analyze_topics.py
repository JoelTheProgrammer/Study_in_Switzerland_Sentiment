import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        print(f"[Topics] No sentiment data found at {input_path}", flush=True)
        print("[Topics] Run analyze_sentiment.py first.", flush=True)
        return

    topic_classifier.load_topic_classifier_config(input_dir)

    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"[Topics] Classifying topics for {total_rows} posts...", flush=True)

    degree_types = []
    main_aspects = []

    progress_every = 50 if total_rows >= 200 else 10

    for i, text in enumerate(df["translated_text"], start=1):
        text = str(text).strip()
        degree_types.append(topic_classifier.get_most_likely_degree(text))
        main_aspects.append(topic_classifier.get_main_aspect(text))

        if i % progress_every == 0 or i == total_rows:
            print(f"[Topics] Processed {i}/{total_rows}", flush=True)

    df["degree_type"] = degree_types
    df["main_aspect"] = main_aspects

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[Topics] Saved topic-annotated data to '{output_path}'", flush=True)


if __name__ == "__main__":
    main()