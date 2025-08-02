import os
import json
from tqdm import tqdm
from models.language import language_detector
from models.translation import translator
from models.qa import topic_classifier

RAW_PATH = "data/raw/raw_posts.json"
PROCESSED_PATH = "data/preprocessed/processed_posts.json"

def enrich_post(post, parent_map):
    """Enrich a single Reddit item (post or comment) with language, translation, and topic classification."""
    text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()

    # Language detection
    lang, conf = language_detector.detect_language(text, return_confidence=True)
    post["lang"] = lang
    post["lang_confidence"] = conf

    # Translation to English
    translated = translator.translate(text, lang)
    post["translated_text"] = translated

    if post["type"] == "post":
        # Classify post normally
        post["is_about_study"] = topic_classifier.is_about_studying_in_switzerland(translated)
        parent_map[post["id"]] = post["is_about_study"]
    else:
        # Comment: inherit classification from its parent post
        parent_id = post.get("post_id")
        post["is_about_study"] = parent_map.get(parent_id, False)

    return post

def main():
    if not os.path.exists(RAW_PATH):
        print("❌ No raw data found. Please run `fetch_posts.py` first.")
        return

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    # Ensure posts are processed before comments
    raw_items.sort(key=lambda x: 0 if x["type"] == "post" else 1)

    print(f"🧠 Processing {len(raw_items)} posts and comments...\n")

    parent_map = {}
    enriched = [enrich_post(item, parent_map) for item in tqdm(raw_items, desc="🔄 Enriching items")]

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(enriched)} enriched items to '{PROCESSED_PATH}'")

if __name__ == "__main__":
    main()
