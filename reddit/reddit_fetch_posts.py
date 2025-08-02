import praw
import json
import os

def fetch_reddit_posts(queries, subreddits, limit=1000):
    reddit = praw.Reddit(
        client_id="wN5a9YGRPB-WPxlGOdS3vQ",
        client_secret="-Ei3oJPinD-K_LGKClYy33e89F9BtA",
        user_agent="StudyInSwitzerlandScraper by u/Additional_Pass7295"
    )

    unique_posts = {}
    seen_users = set()

    for subreddit in subreddits:
        print(f"🔍 Searching r/{subreddit}...")
        for q in queries:
            print(f" → Query: {q}")
            try:
                for post in reddit.subreddit(subreddit).search(q, sort="new", limit=limit):

                    seen_users.add(post.author.name)
                    unique_posts[post.id] = {
                        "id": post.id,
                        "author": post.author.name,
                        "title": post.title,
                        "selftext": post.selftext,
                        "subreddit": post.subreddit.display_name,
                        "query": q,
                        "score": post.score,
                        "url": post.url,
                        "created_utc": post.created_utc
                    }
            except Exception as e:
                print(f"⚠️ Error on query '{q}': {e}")

    # Save to file
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_posts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(unique_posts.values()), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(unique_posts)} raw posts to '{output_path}'")

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

    fetch_reddit_posts(queries, subreddits, limit=2)
