import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from collections import Counter

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

# ======================== LABELS ========================

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

ASPECT_LABELS = [
    "problems with cost of living or tuition fees",
    "location and environment of universities",
    "difficulties with the local language",
    "job opportunities after graduation",
    "quality of professors and teaching staff",
    "no clear aspect mentioned"
]

# ======================== KEYWORDS ========================

DEGREE_KEYWORDS = {
    "bachelor studies": ["bachelor", "undergraduate", "bsc", "ba"],
    "master studies": ["master", "graduate program", "msc", "ma"],
    "phd studies": ["phd", "doctoral", "doctorate", "dphil"],
}

ASPECT_KEYWORDS = {
    "high tuition fees or cost of living": [
        "tuition", "fee", "fees", "expensive", "cost", "costs", "money", "rent", "housing", "afford", "living expenses", "scholarship", "financial"
    ],
    "location and environment of universities": [
        "location", "city", "place", "campus", "environment", "area", "geneva", "lausanne", "zurich", "bern", "switzerland", "transport", "public transport"
    ],
    "difficulties with the local language": [
        "language", "french", "german", "italian", "english", "learn", "speaking", "understand", "communication", "language barrier"
    ],
    "job opportunities after graduation": [
        "job", "jobs", "work", "employment", "career", "internship", "opportunity", "hiring", "find work", "looking for a job", "career prospects"
    ],
    "quality of professors and teaching staff": [
        "teacher", "teachers", "professor", "professors", "lecturer", "lecturers", "teaching", "staff", "class", "courses", "quality of teaching", "bad professor", "good professor"
    ],
}



# ======================== DETECTION FUNCTIONS ========================

def is_about_studying_in_switzerland(text: str, threshold: float = 0.5) -> bool:
    """Checks if the text is about studying in Switzerland using zero-shot classification."""
    result = classifier(text, candidate_labels=CANDIDATE_LABELS)
    for label, score in zip(result["labels"], result["scores"]):
        if label == "studying in Switzerland" and score >= threshold:
            return True
    return False


# ===== DEGREE DETECTION (HYBRID) =====

def is_about_bachelor(text: str, threshold: float = 0.5) -> bool:
    text_lower = text.lower()
    if any(kw in text_lower for kw in DEGREE_KEYWORDS["bachelor studies"]):
        return True
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    return any(label == "bachelor studies" and score >= threshold for label, score in zip(result["labels"], result["scores"]))

def is_about_master(text: str, threshold: float = 0.5) -> bool:
    text_lower = text.lower()
    if any(kw in text_lower for kw in DEGREE_KEYWORDS["master studies"]):
        return True
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    return any(label == "master studies" and score >= threshold for label, score in zip(result["labels"], result["scores"]))

def is_about_phd(text: str, threshold: float = 0.5) -> bool:
    text_lower = text.lower()
    if any(kw in text_lower for kw in DEGREE_KEYWORDS["phd studies"]):
        return True
    result = classifier(text, candidate_labels=DEGREE_LABELS)
    return any(label == "phd studies" and score >= threshold for label, score in zip(result["labels"], result["scores"]))

def get_most_likely_degree(text: str, threshold: float = 0.5) -> str:
    """Returns the most likely degree (bachelor, master, phd) using keywords first, then zero-shot classification."""
    text_lower = text.lower()
    for degree, keywords in DEGREE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return degree

    result = classifier(text, candidate_labels=DEGREE_LABELS)
    return result["labels"][0] if result["scores"][0] >= threshold else "unknown"


# ===== ASPECT DETECTION (HYBRID) =====

def get_main_aspect_mentioned(text: str, threshold: float = 0.2) -> str:
    """
    Detects the main aspect mentioned in text.
    - Step 1: Expanded keyword search on the whole text and per sentence.
    - Step 2: Zero-shot classification on sentences, voting for the most common.
    - Step 3: Returns the aspect with the highest votes or confidence.
    """
    if not text or len(text.strip()) < 5:
        return "unknown"

    text_lower = text.lower()

    # --- Step 1: Keyword search in full text ---
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return aspect

    # --- Step 2: Sentence splitting ---
    sentences = re.split(r"[.!?]\s+", text.strip())
    sentences = [s for s in sentences if len(s) > 5]

    # --- Step 3: Keyword matching per sentence ---
    for s in sentences:
        s_lower = s.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in s_lower for kw in keywords):
                return aspect

    # --- Step 4: Zero-shot classification with voting ---
    votes = Counter()
    best_label = "unknown"
    best_score = 0.0

    for s in sentences:
        result = classifier(s, candidate_labels=ASPECT_LABELS)
        label, score = result["labels"][0], result["scores"][0]

        if score >= threshold:
            votes[label] += score
            if score > best_score:
                best_label, best_score = label, score

    # --- Step 5: Return most voted label ---
    if votes:
        return max(votes, key=votes.get)
    return best_label
