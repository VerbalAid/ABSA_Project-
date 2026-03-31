from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

TRIPLET_SEPARATOR = ";"


def parse_triplet_string(s: str) -> List[Tuple[str, str, str]]:
    """
    Parse a string like:
      "(bathroom, rooms cleanliness, negative); (staff, service general, positive)"
    into a list of (term, category, sentiment).
    """
    s = s.strip()
    if not s:
        return []

    triplets: List[Tuple[str, str, str]] = []
    for chunk in s.split(TRIPLET_SEPARATOR):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk[0] == "(" and chunk[-1] == ")":
            chunk = chunk[1:-1]
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            continue
        term, category, sentiment = parts
        triplets.append((term, category, sentiment))
    return triplets


def _micro_tp_fp_fn(
    gold: Sequence[Iterable[Tuple[str, str, str]]],
    pred: Sequence[Iterable[Tuple[str, str, str]]],
) -> Tuple[int, int, int]:
    """Compute micro tp, fp, fn over exact triplet matches."""
    assert len(gold) == len(pred)
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        g_set = set(g)
        p_set = set(p)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    return tp, fp, fn


def micro_f1(
    gold: Sequence[Iterable[Tuple[str, str, str]]],
    pred: Sequence[Iterable[Tuple[str, str, str]]],
) -> float:
    """Compute micro-F1 over exact triplet matches."""
    tp, fp, fn = _micro_tp_fp_fn(gold, pred)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def micro_precision_recall_f1(
    gold: Sequence[Iterable[Tuple[str, str, str]]],
    pred: Sequence[Iterable[Tuple[str, str, str]]],
) -> Tuple[float, float, float]:
    """Return (micro_precision, micro_recall, micro_f1)."""
    tp, fp, fn = _micro_tp_fp_fn(gold, pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

