from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Model info
model_id = "facebook/bart-large-mnli"
save_dir = "../models/qa/local_models/bart_mnli"

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ BART MNLI model saved to '{save_dir}'")
