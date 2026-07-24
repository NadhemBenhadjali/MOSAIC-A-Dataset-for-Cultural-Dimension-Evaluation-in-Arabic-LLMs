"""Few-shot scenario augmentation utilities."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from .dataset import detect_columns
from .io import read_csv, write_csv
from .translation import build_generation_pipeline


DIMENSION_LABELS = {
    "idv": ("النزعة الفردية", "النزعة الجماعية"),
    "lto": ("الرؤية بعيدة المدى", "الرؤية قصيرة المدى"),
    "mas": ("القيم الذكورية", "القيم الأنثوية"),
    "pdi": ("قبول التفاوت في السلطة", "المساواة في السلطة"),
    "uai": ("النفور من الغموض", "تقبّل الغموض"),
}

CHOICE_PATTERN = re.compile(
    r"(?:الخيار|خيار)\s*(?:الأول|الاول|1|١).*?[:：]\s*(.+?)\s*"
    r"(?:الخيار|خيار)\s*(?:الثاني|2|٢).*?[:：]\s*(.+)",
    flags=re.DOTALL | re.IGNORECASE,
)


def build_augmentation_prompt(
    examples: list[tuple[str, str, str]],
    *,
    dimension: str,
) -> str:
    """Build the few-shot augmentation prompt used to create new scenarios."""

    label_1, label_2 = DIMENSION_LABELS[dimension]
    blocks = []
    for i, (scenario, option_1, option_2) in enumerate(examples, start=1):
        blocks.append(
            f"مثال {i}:\n{scenario}\n<|end|>\n"
            f"الخيار 1 ({label_1}): {option_1}\n"
            f"الخيار 2 ({label_2}): {option_2}"
        )

    examples_block = "\n\n".join(blocks)
    return f"""
البعد الثقافي: {dimension}

فيما يلي أمثلة على مواقف تعبّر عن هذا البعد:

{examples_block}

الرجاء توليد موقف جديد يجسد البعد نفسه.
اكتب فقرة عربية قصيرة من عدة جمل.
استخدم أسماء وأماكن عربية.
لا تضف تحليلات أو ملاحظات أو أقواس توضيحية.
ابدأ بصيغة شخصية مباشرة مثل: أنا أو اسمي.
اكتب الموقف أولًا، ثم <|end|>، ثم الخيارين فقط.
الخيار 1 يجب أن يعكس: {label_1}.
الخيار 2 يجب أن يعكس: {label_2}.

موقف جديد:
""".strip()


def parse_generated_scenario(text: str) -> dict[str, str] | None:
    """Parse one generated scenario into prompt and two options."""

    if "<|end|>" not in text:
        return None

    scenario, choices = text.split("<|end|>", 1)
    match = CHOICE_PATTERN.search(choices)
    if not match:
        return None

    return {
        "Arabic_English_Prompt": scenario.strip(),
        "Arabic_Expected_Option_1": match.group(1).strip(" .،؛\n"),
        "Arabic_Expected_Option_2": match.group(2).strip(" .،؛\n"),
    }


def augment_csv(
    input_file: Path,
    output_file: Path,
    *,
    dimension: str,
    n_generations: int,
    model_name: str = "QCRI/Fanar-1-9B-Instruct",
    max_new_tokens: int = 300,
) -> pd.DataFrame:
    """Generate new MOSAIC-style examples from an existing dimension CSV."""

    df = read_csv(input_file)
    prompt_col, option_1_col, option_2_col = detect_columns(df)
    examples = list(df[[prompt_col, option_1_col, option_2_col]].itertuples(index=False))
    generator = build_generation_pipeline(model_name)

    prompts = [
        build_augmentation_prompt(
            random.sample(examples, 3),
            dimension=dimension,
        )
        for _ in range(n_generations)
    ]
    outputs = generator(prompts, max_new_tokens=max_new_tokens)

    rows = []
    for output in outputs:
        parsed = parse_generated_scenario(output[0]["generated_text"].strip())
        if parsed is not None:
            rows.append(parsed)

    augmented = pd.DataFrame(rows)
    write_csv(augmented, output_file)
    return augmented
