from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import PreTrainedTokenizerBase


TRIPLET_SEPARATOR = "; "


@dataclass
class AbsaConfig:
    dataset_name: str
    max_input_length: int
    max_output_length: int
    language: str = ""
    domain: str = ""
    domains_languages: Optional[List[Tuple[str, str]]] = None  # [(domain, lang), ...] for multi


def _format_triplets(triplets: List[List[str]]) -> str:
    """
    Convert list of [aspect term, aspect category, sentiment] into a string.
    Example:
      [['bathroom', 'rooms cleanliness', 'negative']] ->
      "(bathroom, rooms cleanliness, negative)"
    """
    parts: List[str] = []
    for t in triplets:
        # Some M-ABSA entries may contain extra fields; we keep the first three.
        if len(t) < 3:
            continue
        term, category, sentiment = t[0], t[1], t[2]
        parts.append(f"({term}, {category}, {sentiment})")
    return TRIPLET_SEPARATOR.join(parts)


def format_example(text_and_labels: str) -> Tuple[str, str]:
    """
    M-ABSA examples are stored as:
      "<sentence>####[[...], [...]]"
    We split on '####' and eval the right-hand side to get triplets.
    """
    try:
        sentence, raw_labels = text_and_labels.split("####", maxsplit=1)
    except ValueError:
        # Fallback: treat whole string as sentence with no labels
        return text_and_labels.strip(), ""

    sentence = sentence.strip()
    raw_labels = raw_labels.strip()
    if not raw_labels:
        return sentence, ""

    # raw_labels is a Python literal list of lists, e.g. "[['coffee', 'food quality', 'positive']]"
    try:
        triplets: List[List[str]] = eval(raw_labels)  # noqa: S307 - dataset is trusted
    except Exception:
        triplets = []

    target = _format_triplets(triplets) if triplets else ""
    return sentence, target


def _mabsa_config_name(cfg: AbsaConfig) -> str:
    """HF config name for M-ABSA: domain_language (e.g. hotel_en)."""
    return f"{cfg.domain}_{cfg.language}"


def _load_mabsa_split(cfg: AbsaConfig, split: str):
    """Load one split for language/domain. Tries config name first, else default + filter."""
    config_name = _mabsa_config_name(cfg)
    try:
        return load_dataset(cfg.dataset_name, config_name, split=split)
    except ValueError:
        pass
    # HF M-ABSA only exposes 'default'; load and filter by domain/language.
    ds = load_dataset(cfg.dataset_name, "default", split=split)
    if "domain" in ds.column_names and "language" in ds.column_names:
        ds = ds.filter(
            lambda r: r["domain"] == cfg.domain and r["language"] == cfg.language
        )
    elif "domain" in ds.column_names and "lang" in ds.column_names:
        ds = ds.filter(
            lambda r: r["domain"] == cfg.domain and r["lang"] == cfg.language
        )
    elif "config" in ds.column_names:
        ds = ds.filter(lambda r: r["config"] == config_name)
    else:
        # M-ABSA repo layout: {train,dev,test}.txt (dev = validation). Try with and without data/.
        from huggingface_hub import hf_hub_download

        split_file = "dev.txt" if split == "validation" else f"{split}.txt"
        for prefix in ("", "data/"):
            try:
                path = hf_hub_download(
                    repo_id=cfg.dataset_name,
                    filename=f"{prefix}{cfg.domain}/{cfg.language}/{split_file}",
                    repo_type="dataset",
                )
                ds = load_dataset("text", data_files={split: path})
                return ds[split]
            except Exception:
                continue
        else:
            raise ValueError(
                f"M-ABSA 'default' has columns {ds.column_names}; no domain/language/config. "
                f"Cannot filter to domain=%r language=%r. Try data_files or add filter columns."
                % (cfg.domain, cfg.language)
            ) from None
    return ds


def load_mabsa_split(
    cfg: AbsaConfig,
    split: str,
) -> DatasetDict:
    """Load a specific language/domain split from M-ABSA."""
    return _load_mabsa_split(cfg, split)


def _pairs_from_config(cfg: AbsaConfig) -> List[Tuple[str, str]]:
    """Domain/language pairs to load: either multi or single."""
    if cfg.domains_languages:
        return cfg.domains_languages
    return [(cfg.domain, cfg.language)]


def prepare_tokenised_dataset(
    cfg: AbsaConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Load and tokenise train/validation/test for one or more (domain, language) pairs.
    Returns a dict of split -> tokenised dataset.
    """
    pairs = _pairs_from_config(cfg)
    # Single-config view for _load_mabsa_split (same dataset_name, lengths)
    single = AbsaConfig(
        dataset_name=cfg.dataset_name,
        domain=pairs[0][0],
        language=pairs[0][1],
        max_input_length=cfg.max_input_length,
        max_output_length=cfg.max_output_length,
    )
    def _mabsa_concat(split: str):
        if len(pairs) == 1:
            return _load_mabsa_split(single, split)
        return concatenate_datasets(
            [
                _load_mabsa_split(
                    AbsaConfig(
                        dataset_name=cfg.dataset_name,
                        domain=d,
                        language=l,
                        max_input_length=cfg.max_input_length,
                        max_output_length=cfg.max_output_length,
                    ),
                    split,
                )
                for d, l in pairs
            ]
        )

    raw = DatasetDict(
        {
            "train": _mabsa_concat("train"),
            "validation": _mabsa_concat("validation"),
            "test": _mabsa_concat("test"),
        }
    )

    # Tokenise train/validation/test splits.

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        inputs: List[str] = []
        targets: List[str] = []
        for text in batch["text"]:
            sentence, target = format_example(text)
            inputs.append(f"review: {sentence}")
            targets.append(target if target else "")

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_input_length,
            padding="max_length",
            truncation=True,
        )

        # For mT5 we can reuse the same tokenizer for targets without a special
        # as_target_tokenizer() context (newer Transformers versions removed it).
        labels = tokenizer(
            targets,
            max_length=cfg.max_output_length,
            padding="max_length",
            truncation=True,
        )

        # Normalise to list of lists of Python ints; replace pad with -100 for loss masking.
        # (Tokenizer may return tensors in some contexts; in-place mutation and non-int
        # comparison can corrupt labels or make all positions -100 -> loss 0, grad_norm nan.)
        raw_ids = labels["input_ids"]
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        pad_token_id = tokenizer.pad_token_id
        label_ids = []
        for seq in raw_ids:
            label_ids.append([
                -100 if (int(t) == pad_token_id) else int(t)
                for t in seq
            ])

        model_inputs["labels"] = label_ids
        return model_inputs

    tokenised = {}
    for split_name, ds in raw.items():
        tokenised[split_name] = ds.map(
            preprocess,
            batched=True,
            remove_columns=ds.column_names,
        )

    return tokenised

