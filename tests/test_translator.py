import pytest
from models.translation import translator

# Define acceptable translations for each language input
ACCEPTABLE_TRANSLATIONS = {
    "de": ["This is a German sentence", "That's a German sentence"],
    "fr": ["I want to study in Switzerland", "I want to study in Swiss"],
    "it": ["This is a sentence in Italian", "That's a sentence in Italian"],
}

@pytest.mark.parametrize("text, lang", [
    ("Das ist ein deutscher Satz.", "de"),
    ("Je veux étudier en Suisse.", "fr"),
    ("Questa è una frase in italiano.", "it"),
    ("This is already English.", "en"),
])
def test_translation_accuracy(text, lang):
    result = translator.translate(text, lang)

    if lang == "en":
        assert result == text
    else:
        expected_options = ACCEPTABLE_TRANSLATIONS.get(lang, [])
        assert any(opt.lower() in result.lower() for opt in expected_options), \
            f"❌ Unexpected translation for '{text}' ({lang}): got → '{result}'"
