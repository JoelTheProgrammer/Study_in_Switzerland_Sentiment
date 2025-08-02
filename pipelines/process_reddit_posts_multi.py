# pipelines/process_reddit_posts_threads.py
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.language import language_detector
from models.translation import translator
from models.qa import topic_classifier

RAW_PATH = "data/raw/raw_posts.json"
PROCESSED_PATH = "data/preprocessed/processed_posts.json"

# Shared map to store post classifications
parent_map = {}

def enrich_post(post):
    text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()

    # Language detection
    lang, conf = language_detector.detect_language(text, return_confidence=True)
    post["lang"] = lang
    post["lang_confidence"] = conf

    # Translation to English (if needed)
    translated = translator.translate(text, lang)
    post["translated_text"] = translated

    # For comments, inherit the parent's classification
    if post["type"] == "comment":
        parent_id = post.get("post_id")
        post["is_about_study"] = parent_map.get(parent_id, False)
    else:
        # For posts, determine classification and store it
        is_about = topic_classifier.is_about_studying_in_switzerland(translated)
        post["is_about_study"] = is_about
        parent_map[post["id"]] = is_about

    return post

def main():
    if not os.path.exists(RAW_PATH):
        print("❌ No raw data found. Please run fetch_posts.py first.")
        return

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    # ✅ Ensure posts are processed before comments
    raw_items.sort(key=lambda x: 0 if x["type"] == "post" else 1)

    print(f"🧠 Processing {len(raw_items)} items (posts + comments) using threads...\n")

    enriched = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(enrich_post, item): item for item in raw_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 Enriching items"):
            try:
                enriched.append(future.result())
            except Exception as e:
                print(f"⚠️ Error enriching item: {e}")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(enriched)} enriched items to '{PROCESSED_PATH}'")

if __name__ == "__main__":
    main()
