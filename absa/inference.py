from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import MT5ForConditionalGeneration

from .metrics import parse_triplet_string
from .modeling import load_tokenizer


DEFAULT_CHECKPOINT_DIR = "checkpoints/english_hotels"


def load_model_and_tokenizer(
    checkpoint_dir: str | None = None,
    config: Dict[str, Any] | None = None,
):
    """
    Load a fine-tuned model and tokenizer for inference.
    """
    ckpt_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    tokenizer = load_tokenizer(str(ckpt_dir))
    model = MT5ForConditionalGeneration.from_pretrained(str(ckpt_dir))
    model.eval()

    # Minimal config needed for generation lengths.
    if config is None:
        config = {
            "data": {
                "max_input_length": 192,
                "max_output_length": 96,
            },
            "training": {
                "generation_num_beams": 4,
            },
        }
    return model, tokenizer, config


def predict_triplets(
    text: str,
    model: MT5ForConditionalGeneration,
    tokenizer,
    config: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """
    Run generation on a single review string and return parsed triplets.
    """
    device = model.device
    prompt = f"review: {text.strip()}"
    inputs = tokenizer(
        [prompt],
        max_length=config["data"]["max_input_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": config["data"]["max_output_length"],
        "num_beams": config["training"].get("generation_num_beams", 4),
    }
    with torch.no_grad():  # type: ignore[name-defined]
        generated = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_triplet_string(decoded)


def predict_triplets_batch(
    texts: List[str],
    model: MT5ForConditionalGeneration,
    tokenizer,
    config: Dict[str, Any],
) -> List[List[Tuple[str, str, str]]]:
    """
    Run triplet extraction on multiple review strings. Returns one list of triplets per input.
    """
    if not texts:
        return []

    device = model.device
    prompts = [f"review: {t.strip()}" for t in texts]
    inputs = tokenizer(
        prompts,
        max_length=config["data"]["max_input_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": config["data"]["max_output_length"],
        "num_beams": config["training"].get("generation_num_beams", 4),
    }
    with torch.no_grad():  # type: ignore[name-defined]
        generated = model.generate(**inputs, **gen_kwargs)

    decoded_list = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [parse_triplet_string(d) for d in decoded_list]

