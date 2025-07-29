# import_models/import_longformer.py

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

model_id = "mrm8488/longformer-base-4096-finetuned-squadv2"
save_dir = "../models/longformer/local_model"

os.makedirs(save_dir, exist_ok=True)

print("⬇️ Downloading Longformer QA model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ Longformer QA model saved to {save_dir}")