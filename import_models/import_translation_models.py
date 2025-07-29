from transformers import MarianTokenizer, MarianMTModel
import os

models = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "it": "Helsinki-NLP/opus-mt-it-en"
}

base_dir = "../models/translation/local_models"
os.makedirs(base_dir, exist_ok=True)

for lang, model_id in models.items():
    print(f"⬇️ Downloading translation model for {lang} → en ...")
    model_dir = os.path.join(base_dir, f"{lang}_to_en")

    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)

    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    print(f"✅ Saved to {model_dir}")

print("\n🎉 All translation models downloaded and saved locally.")
