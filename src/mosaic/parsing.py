"""Parse model answers into option numbers."""

from __future__ import annotations

import re
import unicodedata

import numpy as np


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for light matching."""

    text = str(text).translate(ARABIC_DIGITS)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    return text.strip().lower()


def parse_option_number(response: str) -> int | float:
    """Extract option 1 or 2 from a model response."""

    if not isinstance(response, str):
        return np.nan

    text = response.translate(ARABIC_DIGITS).strip()
    match = re.match(r"^([12])([\.\)\:،\s]|$)", text)
    if match:
        return int(match.group(1))

    normalized = normalize_arabic(text)
    first_patterns = [r"\bالخيار\s*الاول\b", r"\bاول\b", r"\bواحد\b"]
    second_patterns = [r"\bالخيار\s*الثاني\b", r"\bثاني\b", r"\bاثنين\b"]

    if any(re.search(pattern, normalized) for pattern in first_patterns):
        return 1
    if any(re.search(pattern, normalized) for pattern in second_patterns):
        return 2

    numbers = re.findall(r"[12]", text)
    if len(numbers) == 1:
        return int(numbers[0])

    return np.nan


def map_to_original_option(
    displayed_option: int | float,
    number_to_original: dict[str, int],
) -> int | float:
    """Map a displayed option number back to the original dataset option."""

    if np.isnan(displayed_option):
        return np.nan
    return number_to_original.get(str(int(displayed_option)), np.nan)
