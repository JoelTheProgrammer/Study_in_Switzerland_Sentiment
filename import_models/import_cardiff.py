from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model info
model_id = "cardiffnlp/twitter-roberta-base-sentiment"
save_dir = "../models/sentiment/local_models/cardiff"

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ Cardiff sentiment model saved to '{save_dir}'")
