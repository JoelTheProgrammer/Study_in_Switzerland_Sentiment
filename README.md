# Reddit Sentiment Pipeline Launcher

This project collects Reddit posts and comments, processes them with local AI models, and creates a final dataset for sentiment and topic analysis.

You can run the full workflow from a simple launcher GUI.

## What the project does

The pipeline runs in this order:

1. **Fetch Reddit posts and comments**
2. **Process posts**
   - language detection
   - translation
   - topic relevance filter
3. **Analyze sentiment**
4. **Analyze topics**
5. **Open visualizer** for charts and filtering

## Project structure

```text
project_root/
  launcher.py
  config/
    install_deps.py
    download_all_models.py
  import_models/
    ... model download scripts ...
  pipelines/
    reddit_fetch_posts_with_comments.py
    process_reddit_posts.py
    analyze_sentiment.py
    analyze_topics.py
  tools/
    full_sentiment_visualizer.py
  data_input/
    study_in_switzerland/
      keywords.json
      subreddits.json
      reddit_api.json
      topic_classifier_config.json
  data_output/
    study_in_switzerland/
      raw/
      preprocessed/
      final/
```

## Requirements

- Python 3.8+
- Internet access for installing packages, downloading models, and fetching Reddit data

## First-time setup

### 1) Install dependencies

```bash
python config/install_deps.py --torch cu121
```

Use `--torch cpu` if you do not use CUDA.

### 2) Download models

```bash
python config/download_all_models.py
```

## Input dataset folder

Each dataset case lives in its own folder inside `data_input/`.

Example:

```text
data_input/study_in_switzerland/
```

Required files in that folder:

- `keywords.json`
- `subreddits.json`
- `reddit_api.json`
- `topic_classifier_config.json`

### Example `reddit_api.json`

```json
{
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET",
  "user_agent": "YourAppName by u/your_username",
  "reddit_search_limit": 15,
  "sleep_seconds": 0.2,
  "min_post_words": 20,
  "min_comment_words": 10,
  "min_post_score": -10
}
```

## Output dataset folder

Each run writes results into a matching folder inside `data_output/`.

Example:

```text
data_output/study_in_switzerland/
  raw/
  preprocessed/
  final/
```

## Run with the launcher

```bash
python launcher.py
```

In the launcher:
1. Pick the input dataset folder
2. Pick the output dataset folder
3. Run setup or pipeline steps with the buttons

## Manual commands

### Fetch Reddit posts
```bash
python pipelines/reddit_fetch_posts_with_comments.py --input-dir data_input/study_in_switzerland --output-dir data_output/study_in_switzerland
```

### Process Reddit posts
```bash
python pipelines/process_reddit_posts.py --input-dir data_input/study_in_switzerland --output-dir data_output/study_in_switzerland
```

### Analyze sentiment
```bash
python pipelines/analyze_sentiment.py --input-dir data_input/study_in_switzerland --output-dir data_output/study_in_switzerland
```

### Analyze topics
```bash
python pipelines/analyze_topics.py --input-dir data_input/study_in_switzerland --output-dir data_output/study_in_switzerland
```

### Open visualizer
```bash
python tools/full_sentiment_visualizer.py --output-dir data_output/study_in_switzerland
```

## Notes

- Topic config loads from the selected input folder.
- Language and sentiment models stay generic.
- You can create multiple cases under `data_input/` and `data_output/`.

## Common issues

- Missing file error: check required JSON filenames in the input folder.
- No raw data found: run the Reddit fetch step first.
- No sentiment data found: run the process and sentiment steps first.
- Model load errors: run the model download script again and confirm local model folders exist.
