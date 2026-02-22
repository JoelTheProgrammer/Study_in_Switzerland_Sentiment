import argparse
import json
from pathlib import Path

from tqdm import tqdm
from models.language import language_detector
from models.translation import translator
from models.qa import topic_classifier


def enrich_post(post, parent_map):
    text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()

    lang, conf = language_detector.detect_language(text, return_confidence=True)
    post["lang"] = lang
    post["lang_confidence"] = conf

    translated = translator.translate(text, lang)
    post["translated_text"] = translated

    if post["type"] == "post":
        post["is_about_study"] = topic_classifier.is_about_main_topic(translated)
        parent_map[post["id"]] = post["is_about_study"]
    else:
        parent_id = post.get("post_id")
        post["is_about_study"] = parent_map.get(parent_id, False)

    return post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    raw_path = output_dir / "raw" / "raw_posts.json"
    processed_path = output_dir / "preprocessed" / "processed_posts.json"

    if not raw_path.exists():
        print(f"No raw data found at {raw_path}")
        print("Run the Reddit fetch script first.")
        return

    topic_classifier.load_topic_classifier_config(input_dir)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    raw_items.sort(key=lambda x: 0 if x.get("type") == "post" else 1)

    print(f"Processing {len(raw_items)} posts and comments...")

    parent_map = {}
    enriched = [enrich_post(item, parent_map) for item in tqdm(raw_items, desc="Enriching items")]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(enriched)} enriched items to '{processed_path}'")


if __name__ == "__main__":
    main()