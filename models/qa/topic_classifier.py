import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load model + tokenizer from local path
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

# Label set
CANDIDATE_LABELS = [
    "studying in Switzerland",
    "living in Switzerland",
    "tourism",
    "food",
    "sports",
    "not about studying"
]

DEGREE_LABELS = [
    "bachelor studies",
    "master studies",
    "phd studies",
    "not about university degrees"
]

def is_about_studying_in_switzerland(text: str, threshold: float = 0.5) -> bool:
    result = classifier(text, candidate_labels=CANDIDATE_LABELS)
    for label, score in zip(result["labels"], result["scores"]):
        if label == "studying in Switzerland" and score >= threshold:
            return True
    return False

def is_about_bachelor(text: str, threshold: float = 0.5) -> bool:
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    for label, score in zip(result["labels"], result["scores"]):
        if label == "bachelor studies" and score >= threshold:
            return True
    return False

def is_about_master(text: str, threshold: float = 0.5) -> bool:
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    for label, score in zip(result["labels"], result["scores"]):
        if label == "master studies" and score >= threshold:
            return True
    return False

def is_about_phd(text: str, threshold: float = 0.5) -> bool:
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    for label, score in zip(result["labels"], result["scores"]):
        if label == "phd studies" and score >= threshold:
            return True
    return False

def get_most_likely_degree(text: str) -> str:
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    return result["labels"][0] if result["scores"][0] >= 0.5 else "unknown"

ASPECT_LABELS = [
    "price",
    "location",
    "language difficulties",
    "job opportunities",
    "teachers",
    "none of these"
]

def get_main_aspect_mentioned(text: str, threshold: float = 0.5) -> str:
    result = classifier(text, candidate_labels=ASPECT_LABELS)
    top_label, top_score = result["labels"][0], result["scores"][0]
    return top_label if top_score >= threshold else "unknown"

