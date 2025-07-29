from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model info
model_id = "j-hartmann/emotion-english-distilroberta-base"
save_dir = "../models/sentiment/local_models/hartmann"

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ Hartmann emotion model saved to '{save_dir}'")
