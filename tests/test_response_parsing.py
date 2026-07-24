import numpy as np

from mosaic.parsing import parse_option_number


def test_parse_ascii_digit():
    assert parse_option_number("1") == 1
    assert parse_option_number("2.") == 2


def test_parse_arabic_digit():
    assert parse_option_number("١") == 1
    assert parse_option_number("٢") == 2


def test_parse_arabic_phrase():
    assert parse_option_number("الخيار الأول") == 1
    assert parse_option_number("الخيار الثاني") == 2


def test_parse_unknown():
    assert np.isnan(parse_option_number("لا أعرف"))
