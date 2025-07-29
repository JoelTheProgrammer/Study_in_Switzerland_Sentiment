import pytest
from models.qa import topic_classifier

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

@pytest.mark.parametrize("text, expected", [
    ("I want to start my bachelor in Switzerland.", True),
    ("Master degree in ETH seems challenging.", False),
    ("PhD options are limited here.", False),
])
def test_is_about_bachelor(text, expected):
    result = topic_classifier.is_about_bachelor(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"

@pytest.mark.parametrize("text, expected", [
    ("I completed my bachelor and am looking for a master program in Switzerland.", True),
    ("Bachelor studies in Geneva are good.", False),
    ("Thinking of doing a PhD after this.", False),
])
def test_is_about_master(text, expected):
    result = topic_classifier.is_about_master(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"

@pytest.mark.parametrize("text, expected", [
    ("PhD in particle physics at EPFL is very demanding.", True),
    ("I just graduated from my master's program.", False),
    ("Looking for bachelor degrees abroad.", False),
])
def test_is_about_phd(text, expected):
    result = topic_classifier.is_about_phd(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"

@pytest.mark.parametrize("text, expected", [
    ("I can’t afford the tuition fees in Switzerland.", "price"),
    ("Geneva has a great international atmosphere.", "location"),
    ("It’s hard to find a job after graduation.", "job opportunities"),
    ("My professor is terrible at explaining things.", "teachers"),
    ("French is hard to learn as a second language.", "language difficulties"),
    ("I like skiing.", "unknown"),
])
def test_get_main_aspect_mentioned(text, expected):
    result = topic_classifier.get_main_aspect_mentioned(text)
    assert result == expected, f"Expected {expected}, got {result} for: {text}"
