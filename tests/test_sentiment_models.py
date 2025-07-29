import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sentiment import hartmann, bert_emotion, cardiff


# === Centralized model registry for optional cross-model tests ===
MODELS = {
    "hartmann": hartmann,
    "bert_emotion": bert_emotion,
    "cardiff": cardiff,
}

# === Test individual model behavior ===

@pytest.mark.parametrize("text, expected_moods", [
    ("I'm so happy I got accepted to ETH Zurich!", ["Positive"]),
    ("Studying abroad makes me anxious.", ["Negative"]),
    ("", ["Neutral"]),  # Edge case: empty input should default to Neutral
])
def test_hartmann_output(text, expected_moods):
    mood = hartmann.classify(text)
    assert mood in expected_moods, f"Hartmann returned {mood} for: {text}"

@pytest.mark.parametrize("text, expected_moods", [
    ("I got a scholarship to EPFL!", ["Positive"]),
    ("Swiss tuition fees are depressing.", ["Negative"]),
    ("", ["Neutral"]),
])
def test_bert_emotion_output(text, expected_moods):
    mood = bert_emotion.classify(text)
    assert mood in expected_moods, f"BERT Emotion returned {mood} for: {text}"

@pytest.mark.parametrize("text, expected_moods", [
    ("ETH is one of the best universities!", ["Positive"]),
    ("I am worried about getting a visa.", ["Negative"]),
    ("", ["Neutral"]),
])
def test_cardiff_output(text, expected_moods):
    mood = cardiff.classify(text)
    assert mood in expected_moods, f"Cardiff returned {mood} for: {text}"

# === Agreement tests on strong sentiment ===

def test_all_models_agree_on_positive():
    text = "I love Swiss universities, they are amazing!"
    for name, model in MODELS.items():
        mood = model.classify(text)
        assert mood == "Positive", f"{name} classified as {mood}, expected Positive"

def test_all_models_agree_on_negative():
    text = "I'm extremely frustrated with the Swiss visa process."
    for name, model in MODELS.items():
        mood = model.classify(text)
        assert mood == "Negative", f"{name} classified as {mood}, expected Negative"
