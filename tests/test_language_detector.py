import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from models.language import language_detector

# === Core 4 Language Coverage ===

@pytest.mark.parametrize("text, expected", [
    # English variants
    ("This is an English sentence.", "en"),
    ("I can't wait to study abroad!", "en"),
    ("My favorite university is in Switzerland.", "en"),

    # German variants
    ("Das ist ein deutscher Satz.", "de"),
    ("Ich möchte in der Schweiz studieren.", "de"),
    ("Die Universitäten dort sind sehr gut.", "de"),

    # French variants
    ("Ceci est une phrase française.", "fr"),
    ("Je veux étudier en Suisse.", "fr"),
    ("L’université est très prestigieuse.", "fr"),

    # Italian variants
    ("Questa è una frase in italiano.", "it"),
    ("Voglio studiare in Svizzera.", "it"),
    ("Le università svizzere sono eccellenti.", "it"),
])
def test_primary_languages(text, expected):
    detected = language_detector.detect_language(text)
    assert detected == expected, f"Expected {expected}, got {detected} for: {text}"

# === Valid foreign languages (optional non-core checks) ===

@pytest.mark.parametrize("text, expected", [
    ("Hola, ¿cómo estás?", "es"),
    ("Это предложение на русском.", "ru"),
    ("这是一段中文。", "zh"),
])
def test_optional_languages(text, expected):
    detected = language_detector.detect_language(text)
    assert detected == expected, f"Expected {expected}, got {detected} for: {text}"

# === Edge cases that should trigger 'unknown' ===

@pytest.mark.parametrize("text", [
    "",                   # empty string
    "     ",              # whitespace only
    "asdfasdfasdf",       # gibberish
    ".....",              # punctuation only
    "1234567890",         # numbers only
])
def test_invalid_input_returns_unknown(text):
    label, confidence = language_detector.detect_language(text, return_confidence=True)
    print(f"⚠️ '{text}' -> Detected: {label}, Confidence: {confidence:.2f}")
    assert label == "unknown", f"Expected 'unknown', got {label} (confidence={confidence:.2f}) for: {text}"

