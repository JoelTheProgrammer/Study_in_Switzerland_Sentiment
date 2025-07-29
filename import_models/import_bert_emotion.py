from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model info
model_id = "bhadresh-savani/bert-base-uncased-emotion"
save_dir = "../models/sentiment/local_models/bert_emotion"

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ Bert Emotion sentiment model saved to '{save_dir}'")