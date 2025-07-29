# tests/test_longformer_qa.py

import pytest
from models.longformer import longformer_qa

@pytest.mark.parametrize("text, expected", [
    ("I'm considering studying at ETH Zurich next year.", True),
    ("Swiss cheese is the best in the world!", False),
    ("What are the visa requirements for studying in Switzerland?", True),
    ("Let's go hiking in the Alps this summer.", False),
    ("Is it expensive to live as a student in Lausanne?", True),
])
def test_is_about_studying(text, expected):
    result = longformer_qa.is_about_studying_in_switzerland(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"
