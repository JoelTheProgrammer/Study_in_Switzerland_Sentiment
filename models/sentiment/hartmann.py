import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Load local model and tokenizer ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_models", "hartmann")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(_device)

# === Raw labels: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'trust']
_LABEL_TO_SENTIMENT = {
    "joy": "Positive",
    "trust": "Positive",
    "neutral": "Neutral",
    "surprise": "Neutral",
    "anger": "Negative",
    "fear": "Negative",
    "sadness": "Negative",
    "disgust": "Negative"
}

def classify(text: str) -> str:
    """Returns: 'Positive', 'Negative', or 'Neutral'."""
    if not text.strip():
        return "Neutral"
    
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        label_id = torch.argmax(probs).item()
        label = _model.config.id2label[label_id].lower()

    return _LABEL_TO_SENTIMENT.get(label, "Neutral")
