import pandas as pd

from mosaic.prompts import create_eval_prompts


def test_prompt_contains_numbered_options():
    df = pd.DataFrame(
        {
            "Arabic_English_Prompt": ["سؤال تجريبي؟"],
            "Arabic_Expected_A": ["الخيار الأول"],
            "Arabic_Expected_B": ["الخيار الثاني"],
        }
    )

    prompts = create_eval_prompts(df, dimension="idv", seed=42)
    prompt = prompts.loc[0, "prompt"]

    assert "1." in prompt
    assert "2." in prompt
    assert "اكتب رقم الخيار فقط" in prompt
