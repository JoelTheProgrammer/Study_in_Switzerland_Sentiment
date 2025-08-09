import time
import praw
import json
import os

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

                    # Fetch and add top-level comments
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
    queries = [
        # English
        "study in switzerland", "university in switzerland", "swiss student visa",
        # German
        "Studium in der Schweiz", "Universität Schweiz", "Studentenvisum Schweiz",
        # French
        "étudier en Suisse", "etudier en suisse", "étudier en suisse pour les africains",
        "étudier médecine en suisse pour les étrangers", "étudier en suisse prix",
        "étudier en suisse pour les sénégalais", "étudier en suisse pour les étrangers",
        "comment faire pour étudier en suisse", "étudier en suisse en français",
        "étudier en suisse avantage", "université suisse", "visa étudiant suisse",
        "universités suisses", "les universités suisses francophones",
        "les universités publiques suisses francophones", "les universités suisses",
        "université de suisse pour étranger",
        # Italian
        "studiare in Svizzera", "studiare in svizzera", "studiare infermieristica in svizzera",
        "studiare medicina in svizzera", "studiare psicologia in svizzera",
        "studiare fisioterapia in svizzera", "studiare in svizzera università",
        "studiare in svizzera costi", "lavorare e studiare in svizzera",
        # Spanish
        "como estudiar en suiza",
        # Related
        "università svizzere", "università telematiche svizzere riconosciute dal miur",
        "migliori università svizzere", "ranking università svizzere",
        "università svizzera", "visto studentesco"
    ]

    subreddits = [
        # Primary subreddits
        "studyAbroad", "university", "Indians_StudyAbroad",
        # Switzerland and region-specific
        "Switzerland", "AskSwitzerland", "Suisse", "svizzera", "ticino",
        "basel", "zurich", "lausanne", "bern",
        # Language/culture
        "de", "france", "italy", "francophonie",
        # Broader
        "AskEurope"
    ]

    fetch_reddit_posts(queries, subreddits, limit=15)

def word_count(text: str) -> int:
    """Count words in a given text."""
    return len(text.strip().split())
