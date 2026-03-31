from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .data import AbsaConfig, prepare_tokenised_dataset
from .metrics import micro_precision_recall_f1, parse_triplet_string
from .modeling import load_mt5, load_tokenizer


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_absa_config(config: Dict[str, Any]) -> AbsaConfig:
    data_cfg = config["data"]
    dl = data_cfg.get("domains_languages")
    if dl:
        pairs = [(d["domain"], d["language"]) for d in dl]
        return AbsaConfig(
            dataset_name=data_cfg["dataset_name"],
            max_input_length=data_cfg["max_input_length"],
            max_output_length=data_cfg["max_output_length"],
            domains_languages=pairs,
        )
    return AbsaConfig(
        dataset_name=data_cfg["dataset_name"],
        language=data_cfg["language"],
        domain=data_cfg["domain"],
        max_input_length=data_cfg["max_input_length"],
        max_output_length=data_cfg["max_output_length"],
    )


def _load_and_tokenise(config: Dict[str, Any], tokenizer) -> DatasetDict:
    absa_cfg = _build_absa_config(config)
    tokenised = prepare_tokenised_dataset(absa_cfg, tokenizer)
    return DatasetDict(tokenised)


def _build_training_arguments(config: Dict[str, Any]) -> Seq2SeqTrainingArguments:
    train_cfg = config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        logging_steps=int(train_cfg.get("logging_steps", 100)),
        predict_with_generate=train_cfg.get("predict_with_generate", True),
        generation_num_beams=int(train_cfg.get("generation_num_beams", 2)),
        # mT5: fp16 causes NaN; bf16/autocast can drop labels → decoder_input_ids error. Use fp32.
        fp16=False,
        bf16=False,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        label_smoothing_factor=float(train_cfg.get("label_smoothing_factor", 0.1)),
    )
    if "optim" in train_cfg:
        kwargs["optim"] = train_cfg["optim"]
    if "eval_strategy" in train_cfg:
        kwargs["eval_strategy"] = train_cfg["eval_strategy"]
    elif "evaluation_strategy" in train_cfg:
        kwargs["eval_strategy"] = train_cfg["evaluation_strategy"]
    if "eval_steps" in train_cfg:
        kwargs["eval_steps"] = int(train_cfg["eval_steps"])
    if "save_steps" in train_cfg:
        kwargs["save_steps"] = int(train_cfg["save_steps"])
    if "save_total_limit" in train_cfg:
        kwargs["save_total_limit"] = int(train_cfg["save_total_limit"])
    if train_cfg.get("dataloader_pin_memory") is False:
        kwargs["dataloader_pin_memory"] = False
    return Seq2SeqTrainingArguments(**kwargs)


def _build_compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        # Convert to numpy int32 to avoid OverflowError in tokenizer.decode (Rust expects int32)
        preds = np.asarray(preds, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        # Clamp preds to valid token range (tokenizer can overflow on bad ids)
        pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
        vocab_size = getattr(tokenizer, "vocab_size", 250112)
        preds = np.clip(preds, 0, vocab_size - 1)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, pad_id)
        labels = np.clip(labels, 0, vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        gold_triplets: List[List[tuple]] = [
            parse_triplet_string(s) for s in decoded_labels
        ]
        pred_triplets: List[List[tuple]] = [
            parse_triplet_string(s) for s in decoded_preds
        ]

        precision, recall, f1 = micro_precision_recall_f1(gold_triplets, pred_triplets)
        return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1}

    return compute_metrics


def run_training(config: Dict[str, Any]) -> None:
    set_seed(config.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_name = config["training"]["model_name"]
    if os.environ.get("USE_MT5_SMALL", "").strip() in ("1", "true", "yes"):
        model_name = "google/mt5-small"
        config["training"]["model_name"] = model_name
    tokenizer = load_tokenizer(model_name)
    model: MT5ForConditionalGeneration = load_mt5(model_name)

    tokenised_ds = _load_and_tokenise(config, tokenizer)
    train_dataset = tokenised_ds["train"]
    eval_dataset = tokenised_ds["validation"]
    train_cfg = config.get("training", {})
    no_eval = train_cfg.get("eval_strategy") == "no"

    if no_eval:
        eval_dataset = None
        del tokenised_ds["validation"], tokenised_ds["test"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args = _build_training_arguments(config)
    compute_metrics = _build_compute_metrics_fn(tokenizer)

    # Seq2SeqTrainer needs DataCollatorForSeq2Seq so batch has decoder_input_ids (from labels).
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # Sanity check: ensure labels have non--100 tokens (otherwise loss=0, grad_norm=nan)
    from torch.utils.data import DataLoader
    check_loader = DataLoader(
        train_dataset,
        batch_size=min(4, len(train_dataset)),
        collate_fn=data_collator,
    )
    first_batch = next(iter(check_loader))
    if "labels" in first_batch:
        lab = first_batch["labels"]
        if torch.is_tensor(lab):
            all_ignore = (lab == -100).all().item()
        else:
            all_ignore = all(
                all(x == -100 for x in seq) for seq in lab
            )
        if all_ignore:
            raise ValueError(
                "All label positions are -100. Check data preprocessing: targets may be empty "
                "or tokenizer pad_token_id masking may be wrong (see absa/data.py)."
            )
    del check_loader, first_batch
    gc.collect()

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


def run_evaluation(config: Dict[str, Any]) -> Dict[str, float]:
    use_cpu = os.environ.get("EVAL_USE_CPU", "").strip().lower() in ("1", "true", "yes")
    if torch.cuda.is_available() and not use_cpu:
        torch.cuda.empty_cache()
    train_cfg = config["training"]
    output_dir = train_cfg["output_dir"]

    tokenizer = load_tokenizer(output_dir)
    model = MT5ForConditionalGeneration.from_pretrained(output_dir)
    if use_cpu:
        model = model.to("cpu")

    tokenised_ds = _load_and_tokenise(config, tokenizer)
    test_dataset = tokenised_ds["test"]

    args = _build_training_arguments(config)
    args.per_device_eval_batch_size = min(args.per_device_eval_batch_size, 2)
    args.generation_num_beams = min(args.generation_num_beams, 2)
    if use_cpu:
        args.per_device_eval_batch_size = 4  # CPU can use a bit larger batch
    compute_metrics = _build_compute_metrics_fn(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(metrics)
    return metrics

