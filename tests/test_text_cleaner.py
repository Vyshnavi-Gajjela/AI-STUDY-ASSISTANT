from src.text_cleaner import clean_text


def test_clean_text_removes_extra_whitespace():
    assert clean_text("  Hello   world \n again  ") == "Hello world again"


def test_clean_text_handles_empty_text():
    assert clean_text("   ") == ""