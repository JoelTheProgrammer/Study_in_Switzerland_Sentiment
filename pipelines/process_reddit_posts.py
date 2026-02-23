import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        print(f"[Process] No raw data found at {raw_path}", flush=True)
        print("[Process] Run the Reddit fetch script first.", flush=True)
        return

    topic_classifier.load_topic_classifier_config(input_dir)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    raw_items.sort(key=lambda x: 0 if x.get("type") == "post" else 1)

    total_items = len(raw_items)
    print(f"[Process] Processing {total_items} posts and comments...", flush=True)

    parent_map = {}
    enriched = []

    progress_every = 50 if total_items >= 200 else 10

    for i, item in enumerate(raw_items, start=1):
        enriched.append(enrich_post(item, parent_map))

        if i % progress_every == 0 or i == total_items:
            print(f"[Process] Processed {i}/{total_items}", flush=True)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"[Process] Saved {len(enriched)} enriched items to '{processed_path}'", flush=True)


if __name__ == "__main__":
    main()