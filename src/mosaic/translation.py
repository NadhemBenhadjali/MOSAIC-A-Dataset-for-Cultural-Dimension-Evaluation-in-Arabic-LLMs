"""Translation and Arabic cultural adaptation helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline

from .io import read_csv, write_csv


DEFAULT_TRANSLATION_MODEL = "QCRI/Fanar-1-9B-Instruct"
END_TAG = "<|end|>"

TRANSLATION_PROMPT = """
ترجم النص التالي إلى العربية فقط، مع تعديل الأسماء والأماكن والمراجع
الثقافية لتكون ملائمة للثقافة العربية. لا تضف أي تفسير أو مثال أو
تعليمات أو جملة إضافية. اكتب الترجمة مباشرة في جملة واضحة ومفهومة.

مثال:
النص: My name is Volodymyr and I am a junior programmer in Kyiv.
الترجمة: اسمي وليد وأنا مبرمج مبتدئ في عمّان.

النص: {text}
الترجمة:
""".strip()


def clean_translation(text: str) -> str:
    """Remove common extra text from a generated translation."""

    cutoff_patterns = [
        re.escape(END_TAG),
        r"\bملاحظة\b",
        r"\bمثال\b",
        r"\bExplanation\b",
        r"\bNote\b",
    ]
    text = re.split("|".join(cutoff_patterns), text, maxsplit=1)[0]
    text = re.sub(r"^\s*الترجمة\s*[:：\-–]*\s*", "", text)
    return text.strip(" \t\r\n\u200e\u200f")


def build_generation_pipeline(model_name: str = DEFAULT_TRANSLATION_MODEL):
    """Load a text-generation pipeline for translation."""

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,
        do_sample=False,
    )


def translate_texts(
    texts: Sequence[str],
    generator,
    *,
    max_new_tokens: int = 300,
) -> list[str]:
    """Translate a batch of English texts into Arabic."""

    prompts = [TRANSLATION_PROMPT.format(text=text) for text in texts]
    outputs = generator(prompts, max_new_tokens=max_new_tokens)
    return [clean_translation(item[0]["generated_text"]) for item in outputs]


def translate_csv(
    input_file: Path,
    output_file: Path,
    *,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
) -> pd.DataFrame:
    """Translate all columns in one CSV file and save Arabic-prefixed columns."""

    df = read_csv(input_file)
    generator = build_generation_pipeline(model_name)
    translated = df.copy()

    for column in df.columns:
        translated[f"Arabic_{column}"] = translate_texts(
            df[column].astype(str).tolist(),
            generator,
        )

    write_csv(translated, output_file)
    return translated
