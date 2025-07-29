from transformers import MarianTokenizer, MarianMTModel
import torch
import os

SUPPORTED_LANGUAGES = ["de", "fr", "it"]

def _resolve_path(relative_path):
    base = os.path.dirname(__file__)
    full_path = os.path.abspath(os.path.join(base, relative_path))
    return full_path.replace("\\", "/")  # Normalize for Hugging Face

# Load models into memory
_models = {}
for lang in SUPPORTED_LANGUAGES:
    model_path = _resolve_path(f"local_models/{lang}_to_en")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"❌ Translation model not found at {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
    _models[lang] = (tokenizer, model)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for _, (_, model) in _models.items():
    model.to(_device)

def translate(text: str, src_lang: str) -> str:
    """
    Translates input text from src_lang ('de', 'fr', 'it') to English.
    If text is already in English or unsupported lang, returns input.
    """
    if src_lang not in SUPPORTED_LANGUAGES:
        return text  # Skip if English or unsupported

    tokenizer, model = _models[src_lang]
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return output
