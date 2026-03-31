from __future__ import annotations

from transformers import MT5ForConditionalGeneration, AutoTokenizer


def load_mt5(model_name: str) -> MT5ForConditionalGeneration:
    return MT5ForConditionalGeneration.from_pretrained(model_name)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad token is set for generation and loss masking
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

