import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_models", "xlm_roberta")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(_device)

def detect_language(text: str, threshold: float = 0.8, return_confidence: bool = False):
    """
    Returns ISO 639-1 code or 'unknown' if confidence is below threshold.
    If return_confidence=True, returns a tuple: (label, confidence)
    """
    if not text.strip():
        return ("unknown", 0.0) if return_confidence else "unknown"
    
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        confidence = torch.max(probs).item()
        label_id = torch.argmax(probs).item()
        label = _model.config.id2label[label_id].replace("__label__", "")

    if return_confidence:
        return (label if confidence >= threshold else "unknown", confidence)
    return label if confidence >= threshold else "unknown"
