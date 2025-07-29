# models/longformer/longformer_qa.py

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os

_model_path = os.path.join(os.path.dirname(__file__), "local_model")
_tokenizer = AutoTokenizer.from_pretrained(_model_path)
_model = AutoModelForQuestionAnswering.from_pretrained(_model_path)
_model.to("cuda" if torch.cuda.is_available() else "cpu")
_device = next(_model.parameters()).device

def is_about_studying_in_switzerland(post: str, threshold: float = 0.0) -> bool:
    question = "Is this post about studying in Switzerland?"

    inputs = _tokenizer(question, post, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    score = ((start_scores[0, start_idx] + end_scores[0, end_idx]) / 2).item()

    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = _tokenizer.decode(answer_tokens, skip_special_tokens=True)

    print(f"📌 Answer: '{answer}' | Score: {score:.2f}")

    return score > threshold and answer.strip() != ""
