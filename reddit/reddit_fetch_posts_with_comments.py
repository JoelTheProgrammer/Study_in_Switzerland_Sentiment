import time
import praw
import json
import os

def load_json_list(file_path):
    """Load a flattened list from a JSON file (which may contain categories)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Flatten all category lists
        items = []
        for values in data.values():
            items.extend(values)
        return sorted(set(items))  # unique, sorted
    elif isinstance(data, list):
        return sorted(set(data))
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")

def word_count(text: str) -> int:
    """Count words in a given text."""
    return len(text.strip().split())

def fetch_reddit_posts(queries, subreddits, limit=1000):
    reddit = praw.Reddit(
        client_id="wN5a9YGRPB-WPxlGOdS3vQ",
        client_secret="-Ei3oJPinD-K_LGKClYy33e89F9BtA",
        user_agent="StudyInSwitzerlandScraper by u/Additional_Pass7295"
    )

    seen_post_keys = set()
    all_items = []

    for subreddit in subreddits:
        print(f"🔍 Searching r/{subreddit}...")
        for q in queries:
            print(f" → Query: {q}")
            try:
                for post in reddit.subreddit(subreddit).search(q, sort="new", limit=limit):
                    if not post.author or post.score < -10:
                        continue

                    # Skip posts with too few words (title excluded)
                    post_word_count = word_count(post.selftext)
                    if post_word_count < 20:
                        continue  # Skip post and its comments entirely

                    post_key = (post.title.strip(), post.selftext.strip())
                    if post_key in seen_post_keys:
                        continue
                    seen_post_keys.add(post_key)

                    # Add post
                    post_data = {
                        "id": post.id,
                        "author": post.author.name,
                        "title": post.title,
                        "selftext": post.selftext,
                        "subreddit": post.subreddit.display_name,
                        "query": q,
                        "score": post.score,
                        "url": post.url,
                        "created_utc": post.created_utc,
                        "type": "post" 
                    }
                    all_items.append(post_data)

                    # Fetch and add top-level comments (only if post passed)
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        if not comment.author:
                            continue

                        # Skip comments with too few words
                        if word_count(comment.body) < 10:
                            continue

                        comment_data = {
                            "id": comment.id,
                            "post_id": post.id,
                            "author": comment.author.name,
                            "title": "",
                            "selftext": comment.body,
                            "subreddit": post.subreddit.display_name,
                            "query": q,
                            "score": comment.score,
                            "url": f"https://reddit.com{comment.permalink}",
                            "created_utc": comment.created_utc,
                            "type": "comment"
                        }
                        all_items.append(comment_data)

            except Exception as e:
                print(f"⚠️ Error on query '{q}' in r/{subreddit}: {e}")

    # Save to file
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_posts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(all_items)} items (posts + comments) to '{output_path}'")

if __name__ == "__main__":
    # Load from JSON files
    keywords_path = os.path.join(os.path.dirname(__file__), "keywords.json")
    subreddits_path = os.path.join(os.path.dirname(__file__), "subreddits.json")

    queries = load_json_list(keywords_path)
    subreddits = load_json_list(subreddits_path)

    print(f"Loaded {len(queries)} search queries and {len(subreddits)} subreddits.")
    fetch_reddit_posts(queries, subreddits, limit=15)
