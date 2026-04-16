from __future__ import annotations

import math

import numpy as np


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, *, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if labels.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(labels.size)
    ece = 0.0
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (probs >= left) & (probs < right if idx < bins - 1 else probs <= right)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += abs(conf - acc) * (float(mask.sum()) / total)
    return ece


def fit_temperature_for_binary_logits(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if labels.size == 0:
        return {"temperature": 1.0, "ece_before": 0.0, "ece_after": 0.0}

    best_temperature = 1.0
    best_nll = math.inf
    for temperature in np.linspace(0.5, 3.0, 26):
        scaled = logits / float(temperature)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        nll = -float(np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)

    before_probs = 1.0 / (1.0 + np.exp(-logits))
    after_probs = 1.0 / (1.0 + np.exp(-(logits / best_temperature)))
    return {
        "temperature": best_temperature,
        "ece_before": expected_calibration_error(labels, before_probs),
        "ece_after": expected_calibration_error(labels, after_probs),
    }
