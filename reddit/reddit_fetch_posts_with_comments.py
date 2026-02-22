import argparse
import json
import time
from pathlib import Path

import praw


def load_json_list(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = []
        for values in data.values():
            items.extend(values)
        items = sorted(set(items))
    elif isinstance(data, list):
        items = sorted(set(data))
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")

    if limit is not None:
        return items[:limit]
    return items


def load_reddit_api_config(input_dir: Path):
    api_path = input_dir / "reddit_api.json"
    if not api_path.exists():
        raise FileNotFoundError(
            f"Missing reddit_api.json in {input_dir}\n"
            f"Create a file with client_id, client_secret, and user_agent."
        )

    with open(api_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["client_id", "client_secret", "user_agent"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"reddit_api.json is missing required keys: {missing}")

    cfg.setdefault("reddit_search_limit", 15)
    cfg.setdefault("sleep_seconds", 0.2)
    cfg.setdefault("min_post_words", 20)
    cfg.setdefault("min_comment_words", 10)
    cfg.setdefault("min_post_score", -10)
    cfg.setdefault("progress_every_n_items", 100)

    return cfg


def word_count(text: str) -> int:
    return len(text.strip().split())


def fetch_reddit_posts(queries, subreddits, output_dir: Path, reddit_cfg):
    reddit = praw.Reddit(
        client_id=reddit_cfg["client_id"],
        client_secret=reddit_cfg["client_secret"],
        user_agent=reddit_cfg["user_agent"],
    )

    search_limit = int(reddit_cfg["reddit_search_limit"])
    sleep_seconds = float(reddit_cfg["sleep_seconds"])
    min_post_words = int(reddit_cfg["min_post_words"])
    min_comment_words = int(reddit_cfg["min_comment_words"])
    min_post_score = int(reddit_cfg["min_post_score"])
    progress_every_n_items = max(1, int(reddit_cfg["progress_every_n_items"]))

    seen_post_keys = set()
    all_items = []

    total_subreddits = len(subreddits)
    total_queries = len(queries)
    total_pairs = total_subreddits * total_queries
    pair_counter = 0

    for subreddit_idx, subreddit in enumerate(subreddits, start=1):
        print(f"[Reddit] Subreddit {subreddit_idx}/{total_subreddits}: r/{subreddit}", flush=True)

        for query_idx, q in enumerate(queries, start=1):
            pair_counter += 1
            print(
                f"[Reddit] Query {query_idx}/{total_queries} in r/{subreddit} "
                f"(overall {pair_counter}/{total_pairs}): {q}",
                flush=True,
            )

            try:
                for post in reddit.subreddit(subreddit).search(q, sort="new", limit=search_limit):
                    if not post.author or post.score < min_post_score:
                        continue

                    if word_count(post.selftext) < min_post_words:
                        continue

                    post_key = (post.title.strip(), post.selftext.strip())
                    if post_key in seen_post_keys:
                        continue
                    seen_post_keys.add(post_key)

                    all_items.append(
                        {
                            "id": post.id,
                            "author": post.author.name,
                            "title": post.title,
                            "selftext": post.selftext,
                            "subreddit": post.subreddit.display_name,
                            "query": q,
                            "score": post.score,
                            "url": post.url,
                            "created_utc": post.created_utc,
                            "type": "post",
                        }
                    )

                    if len(all_items) % progress_every_n_items == 0:
                        print(f"[Reddit] Collected {len(all_items)} items so far...", flush=True)

                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        if not comment.author:
                            continue
                        if word_count(comment.body) < min_comment_words:
                            continue

                        all_items.append(
                            {
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
                                "type": "comment",
                            }
                        )

                        if len(all_items) % progress_every_n_items == 0:
                            print(f"[Reddit] Collected {len(all_items)} items so far...", flush=True)

            except Exception as e:
                print(f"[Reddit] Error on query '{q}' in r/{subreddit}: {e}", flush=True)

            time.sleep(sleep_seconds)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "raw_posts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    post_count = sum(1 for item in all_items if item.get("type") == "post")
    comment_count = sum(1 for item in all_items if item.get("type") == "comment")

    print(
        f"[Reddit] Saved {len(all_items)} items ({post_count} posts + {comment_count} comments) to '{output_path}'",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--subreddit-limit", type=int, default=None)
    parser.add_argument("--reddit-search-limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    keywords_path = input_dir / "keywords.json"
    subreddits_path = input_dir / "subreddits.json"

    if not keywords_path.exists():
        raise FileNotFoundError(f"Missing file: {keywords_path}")
    if not subreddits_path.exists():
        raise FileNotFoundError(f"Missing file: {subreddits_path}")

    queries = load_json_list(keywords_path, limit=args.query_limit)
    subreddits = load_json_list(subreddits_path, limit=args.subreddit_limit)
    reddit_cfg = load_reddit_api_config(input_dir)

    if args.reddit_search_limit is not None:
        reddit_cfg["reddit_search_limit"] = args.reddit_search_limit

    print(f"[Reddit] Loaded {len(queries)} queries and {len(subreddits)} subreddits.", flush=True)

    fetch_reddit_posts(
        queries=queries,
        subreddits=subreddits,
        output_dir=output_dir,
        reddit_cfg=reddit_cfg,
    )

if __name__ == "__main__":
    main()