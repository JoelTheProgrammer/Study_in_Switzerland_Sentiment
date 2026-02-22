import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_models", "bart_mnli")
_device = 0 if torch.cuda.is_available() else -1

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

classifier = pipeline(
    "zero-shot-classification",
    model=_model,
    tokenizer=_tokenizer,
    device=_device
)

_CONFIG = {
    "main_topic_label": "",
    "candidate_labels": [],
    "degree_labels": [],
    "aspect_labels": [],
    "degree_keywords": {},
    "aspect_keywords": {},
}


def load_topic_classifier_config(input_dir: str | Path) -> None:
    input_dir = Path(input_dir)
    cfg_path = input_dir / "topic_classifier_config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = [
        "main_topic_label",
        "candidate_labels",
        "degree_labels",
        "aspect_labels",
        "degree_keywords",
        "aspect_keywords",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"topic_classifier_config.json missing keys: {missing}")

    _CONFIG["main_topic_label"] = cfg["main_topic_label"]
    _CONFIG["candidate_labels"] = cfg["candidate_labels"]
    _CONFIG["degree_labels"] = cfg["degree_labels"]
    _CONFIG["aspect_labels"] = cfg["aspect_labels"]
    _CONFIG["degree_keywords"] = cfg["degree_keywords"]
    _CONFIG["aspect_keywords"] = cfg["aspect_keywords"]


def _ensure_config_loaded() -> None:
    if not _CONFIG["candidate_labels"]:
        raise RuntimeError(
            "Topic classifier config is not loaded. "
            "Call load_topic_classifier_config(input_dir) before using classifier functions."
        )


def is_about_main_topic(text: str, threshold: float = 0.5) -> bool:
    _ensure_config_loaded()

    result = classifier(text, candidate_labels=_CONFIG["candidate_labels"])
    main_label = _CONFIG["main_topic_label"]

    for label, score in zip(result["labels"], result["scores"]):
        if label == main_label and score >= threshold:
            return True
    return False


def is_about_degree(text: str, degree_label: str, threshold: float = 0.5) -> bool:
    _ensure_config_loaded()

    text_lower = text.lower()
    degree_keywords = _CONFIG["degree_keywords"]

    if degree_label in degree_keywords and any(kw in text_lower for kw in degree_keywords[degree_label]):
        return True

    result = classifier(text, candidate_labels=_CONFIG["degree_labels"])
    return any(
        label == degree_label and score >= threshold
        for label, score in zip(result["labels"], result["scores"])
    )


def get_most_likely_degree(text: str, threshold: float = 0.5) -> str:
    _ensure_config_loaded()

    text_lower = text.lower()

    for degree_label, keywords in _CONFIG["degree_keywords"].items():
        if any(kw in text_lower for kw in keywords):
            return degree_label

    result = classifier(text, candidate_labels=_CONFIG["degree_labels"])
    if result["scores"][0] >= threshold:
        return result["labels"][0]
    return "unknown"


def get_main_aspect(text: str, threshold: float = 0.2) -> str:
    _ensure_config_loaded()

    if not text or len(text.strip()) < 5:
        return "unknown"

    text_lower = text.lower()
    aspect_keywords = _CONFIG["aspect_keywords"]

    for aspect_label, keywords in aspect_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return aspect_label

    sentences = re.split(r"[.!?]\s+", text.strip())
    sentences = [s for s in sentences if len(s) > 5]

    for sentence in sentences:
        s_lower = sentence.lower()
        for aspect_label, keywords in aspect_keywords.items():
            if any(kw in s_lower for kw in keywords):
                return aspect_label

    votes = Counter()
    best_label = "unknown"
    best_score = 0.0

    for sentence in sentences:
        result = classifier(sentence, candidate_labels=_CONFIG["aspect_labels"])
        label = result["labels"][0]
        score = result["scores"][0]

        if score >= threshold:
            votes[label] += score
            if score > best_score:
                best_label = label
                best_score = score

    if votes:
        return max(votes, key=votes.get)
    return best_label


def get_topic_labels() -> list[str]:
    _ensure_config_loaded()
    return list(_CONFIG["candidate_labels"])


def get_degree_labels() -> list[str]:
    _ensure_config_loaded()
    return list(_CONFIG["degree_labels"])


def get_aspect_labels() -> list[str]:
    _ensure_config_loaded()
    return list(_CONFIG["aspect_labels"])


# Optional compatibility wrappers so old code keeps working during migration

def is_about_studying_in_switzerland(text: str, threshold: float = 0.5) -> bool:
    return is_about_main_topic(text, threshold=threshold)


def is_about_bachelor(text: str, threshold: float = 0.5) -> bool:
    return is_about_degree(text, "bachelor studies", threshold=threshold)


def is_about_master(text: str, threshold: float = 0.5) -> bool:
    return is_about_degree(text, "master studies", threshold=threshold)


def is_about_phd(text: str, threshold: float = 0.5) -> bool:
    return is_about_degree(text, "phd studies", threshold=threshold)


def get_main_aspect_mentioned(text: str, threshold: float = 0.2) -> str:
    return get_main_aspect(text, threshold=threshold)