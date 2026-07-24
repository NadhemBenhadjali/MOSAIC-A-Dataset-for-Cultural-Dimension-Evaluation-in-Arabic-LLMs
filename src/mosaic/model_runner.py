"""Small Hugging Face runner used by the evaluation scripts."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .constants import MODEL_REPOS


DEFAULT_QUANTIZATION = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def login_if_token_exists() -> None:
    """Log in to Hugging Face when HF_TOKEN is available."""

    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)


@lru_cache(maxsize=8)
def load_model(model_alias: str) -> tuple[Any, Any]:
    """Load and cache a tokenizer/model pair."""

    login_if_token_exists()
    repo_id = MODEL_REPOS.get(model_alias, model_alias)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=DEFAULT_QUANTIZATION,
        torch_dtype=None,
    )
    return tokenizer, model


def set_seed(seed: int) -> None:
    """Best-effort reproducibility for local generation."""

    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_response(
    prompt: str,
    *,
    model_alias: str,
    seed: int | None = None,
    max_new_tokens: int = 24,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate one model response for one prompt."""

    if seed is not None:
        set_seed(seed)

    tokenizer, model = load_model(model_alias)
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
