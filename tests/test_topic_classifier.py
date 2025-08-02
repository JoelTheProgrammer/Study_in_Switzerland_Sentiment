import pytest
from models.qa import topic_classifier

# ===== Tests for studying in Switzerland =====
@pytest.mark.parametrize("text, expected", [
    ("I'm considering studying at ETH Zurich next year.", True),
    ("Swiss cheese is the best in the world!", False),
    ("What are the visa requirements for studying in Switzerland?", True),
    ("Let's go hiking in the Alps this summer.", False),
    ("Is it expensive to live as a student in Lausanne?", True),
    ("I love Rösti and other Swiss dishes.", False),
])
def test_is_about_studying(text, expected):
    result = topic_classifier.is_about_studying_in_switzerland(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"


# ===== Tests for bachelor detection =====
@pytest.mark.parametrize("text, expected", [
    ("I want to start my bachelor in Switzerland.", True),
    ("Master degree in ETH seems challenging.", False),
    ("PhD options are limited here.", False),
    ("Looking for undergraduate programs in Zurich.", True),  # new keyword case
])
def test_is_about_bachelor(text, expected):
    result = topic_classifier.is_about_bachelor(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"


# ===== Tests for master detection =====
@pytest.mark.parametrize("text, expected", [
    ("I completed my bachelor and am looking for a master program in Switzerland.", True),
    ("Bachelor studies in Geneva are good.", False),
    ("Thinking of doing a PhD after this.", False),
    ("Considering an MSc program at EPFL.", True),  # new keyword case
])
def test_is_about_master(text, expected):
    result = topic_classifier.is_about_master(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"


# ===== Tests for PhD detection =====
@pytest.mark.parametrize("text, expected", [
    ("PhD in particle physics at EPFL is very demanding.", True),
    ("I just graduated from my master's program.", False),
    ("Looking for bachelor degrees abroad.", False),
    ("I am applying for a doctoral program in Switzerland.", True),  # new keyword case
])
def test_is_about_phd(text, expected):
    result = topic_classifier.is_about_phd(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"


# ===== Tests for aspect detection =====
@pytest.mark.parametrize("text, expected", [
    ("I can’t afford the tuition fees in Switzerland.", "high tuition fees or cost of living"),
    ("Geneva has a great international atmosphere.", "location and environment of universities"),
    ("It’s hard to find a job after graduation.", "job opportunities after graduation"),
    ("My professor is terrible at explaining things.", "quality of professors and teaching staff"),
    ("French is hard to learn as a second language.", "difficulties with the local language"),
    ("I like skiing.", "unknown"),
    ("Housing costs in Zurich are too high for students.", "high tuition fees or cost of living"),  # keyword match
])
def test_get_main_aspect_mentioned(text, expected):
    result = topic_classifier.get_main_aspect_mentioned(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"
