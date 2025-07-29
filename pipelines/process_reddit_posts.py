import os
import json
from tqdm import tqdm
from models.language import language_detector
from models.translation import translator
from models.qa import topic_classifier

RAW_PATH = "data/raw/raw_posts.json"
PROCESSED_PATH = "data/preprocessed/processed_posts.json"

def enrich_post(post):
    """Enrich a single Reddit post with language, translation, and topic classification."""
    text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()

    # Language detection
    lang, conf = language_detector.detect_language(text, return_confidence=True)
    post["lang"] = lang
    post["lang_confidence"] = conf

    # Translation to English (if needed)
    translated = translator.translate(text, lang)
    post["translated_text"] = translated

    # Determine if it's about studying in Switzerland
    post["is_about_study"] = topic_classifier.is_about_studying_in_switzerland(translated)

    return post

def main():
    if not os.path.exists(RAW_PATH):
        print("❌ No raw data found. Please run `fetch_posts.py` first.")
        return

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_posts = json.load(f)
        #raw_posts = [p for p in raw_posts if p["type"] == "post"]  # or "comment"


    print(f"🧠 Processing {len(raw_posts)} posts...\n")

    # Use tqdm for progress bar
    enriched = [enrich_post(post) for post in tqdm(raw_posts, desc="🔄 Enriching posts")]

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(enriched)} enriched posts to '{PROCESSED_PATH}'")

if __name__ == "__main__":
    main()
