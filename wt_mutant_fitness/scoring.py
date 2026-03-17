import numpy as np
from typing import Optional

BASE_TO_ASCII = {"A": 65, "C": 67, "G": 71, "T": 84}
ASCII_TO_BASE = {v: k for k, v in BASE_TO_ASCII.items()}


def tokenize(sequence: str) -> np.ndarray:
    """Convert a DNA string (ACGT) to an array of ASCII token IDs."""
    seq = sequence.upper().strip()
    ids = []
    for ch in seq:
        if ch not in BASE_TO_ASCII:
            raise ValueError(f"Invalid nucleotide '{ch}'; only A/C/G/T are accepted.")
        ids.append(BASE_TO_ASCII[ch])
    return np.array(ids, dtype=np.int64)


def logits_to_logprobs(logits: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Compute per-position next-token log-probabilities.

    logits:    shape [L, vocab_size] (squeezed from forward output)
    token_ids: shape [L]

    Returns: shape [L-1] array of log P(token_{i+1} | token_{1..i})
    """
    pred_logits = logits[:-1]  # predict next token
    targets = token_ids[1:]

    log_softmax = pred_logits - np.logaddexp.reduce(pred_logits, axis=-1, keepdims=True)
    per_pos = log_softmax[np.arange(len(targets)), targets]
    return per_pos


def score_sequence(logits: np.ndarray, token_ids: np.ndarray, method: str = "mean") -> float:
    """Scalar fitness score: mean (or sum) of per-position log-probs."""
    lp = logits_to_logprobs(logits, token_ids)
    if method == "mean":
        return float(np.mean(lp))
    elif method == "sum":
        return float(np.sum(lp))
    raise ValueError(f"Unknown method '{method}'")


def per_position_scores(logits: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Return the full per-position log-probability array (length L-1)."""
    return logits_to_logprobs(logits, token_ids)


def extract_logits(forward_result: dict) -> np.ndarray:
    """Pull out the logits matrix [L, 512] from the forward() response dict."""
    raw = forward_result["logits"]
    if raw.ndim == 3:
        raw = raw[0]  # (1, L, 512) -> (L, 512)
    return raw.astype(np.float64)


def find_mutations(wt: str, mut: str) -> list[dict]:
    """Compare two equal-length sequences and return list of substitution dicts."""
    wt_u, mut_u = wt.upper(), mut.upper()
    diffs = []
    for i, (w, m) in enumerate(zip(wt_u, mut_u)):
        if w != m:
            diffs.append({"pos": i, "ref": w, "alt": m})
    return diffs


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]
