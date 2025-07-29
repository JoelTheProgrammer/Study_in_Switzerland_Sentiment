from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "papluca/xlm-roberta-base-language-detection"
save_dir = "../models/language/local_models/xlm_roberta"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ Language detection model saved to: {save_dir}")
