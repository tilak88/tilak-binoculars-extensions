"""Pure numerical feature primitives used by the project.

The functions operate on aligned next-token probability arrays. Keeping this
layer framework-neutral makes the scoring logic testable without downloading
multi-billion-parameter models.
"""

from __future__ import annotations

import numpy as np


def _validate_logits(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError("logits must have shape (tokens, vocabulary)")
    return x


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    return logits - _logsumexp(logits)[:, None]


def cross_entropy_sequence(logits: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Return per-token negative log probability for the observed tokens."""
    x = _validate_logits(logits)
    ids = np.asarray(token_ids, dtype=int)
    if ids.ndim != 1 or len(ids) != len(x):
        raise ValueError("token_ids must align with logits")
    if np.any(ids < 0) or np.any(ids >= x.shape[1]):
        raise ValueError("token id outside vocabulary")
    return -_log_softmax(x)[np.arange(len(ids)), ids]


def binoculars_score(observer_logits: np.ndarray, performer_logits: np.ndarray,
                     token_ids: np.ndarray) -> float:
    """Compute observer perplexity divided by observer/performer cross-PPL."""
    obs = _validate_logits(observer_logits)
    perf = _validate_logits(performer_logits)
    if obs.shape != perf.shape:
        raise ValueError("observer and performer logits must have the same shape")
    obs_logp = _log_softmax(obs)
    perf_logp = _log_softmax(perf)
    ids = np.asarray(token_ids, dtype=int)
    if len(ids) != len(obs):
        raise ValueError("token_ids must align with logits")
    ppl = np.exp(np.mean(-obs_logp[np.arange(len(ids)), ids]))
    # Cross entropy H(p_observer, q_performer).
    p = np.exp(obs_logp)
    cross_ppl = np.exp(np.mean(-(p * perf_logp).sum(axis=1)))
    return float(ppl / max(cross_ppl, np.finfo(float).tiny))


def roughness(performer_logits: np.ndarray, token_ids: np.ndarray) -> float:
    return float(np.std(cross_entropy_sequence(performer_logits, token_ids)))


def token_agreement_rate(observer_logits: np.ndarray, performer_logits: np.ndarray) -> float:
    obs = _validate_logits(observer_logits)
    perf = _validate_logits(performer_logits)
    if obs.shape != perf.shape:
        raise ValueError("logit arrays must have the same shape")
    return float(np.mean(np.argmax(obs, axis=1) == np.argmax(perf, axis=1)))


def _ranks(probabilities: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=int)
    if ids.ndim != 1 or len(ids) != len(probabilities):
        raise ValueError("token_ids must align with probabilities")
    if np.any(ids < 0) or np.any(ids >= probabilities.shape[1]):
        raise ValueError("token id outside vocabulary")
    chosen = probabilities[np.arange(len(ids)), ids]
    return np.sum(probabilities > chosen[:, None], axis=1).astype(float)


def rank_volatility(performer_logits: np.ndarray, token_ids: np.ndarray,
                    window: int = 15) -> float:
    perf = np.exp(_log_softmax(_validate_logits(performer_logits)))
    ids = np.asarray(token_ids, dtype=int)
    ranks = _ranks(perf, ids) / perf.shape[1]
    if len(ranks) < 2:
        return 0.0
    w = min(max(2, window), len(ranks))
    return float(np.mean([np.std(ranks[i:i + w]) for i in range(len(ranks) - w + 1)]))


def surprisal_trajectory_curvature(performer_logits: np.ndarray,
                                   token_ids: np.ndarray) -> float:
    ce = cross_entropy_sequence(performer_logits, token_ids)
    if len(ce) < 3:
        return 0.0
    return float(np.var(np.diff(ce, n=2)))


def cmrd_variance(observer_logits: np.ndarray, performer_logits: np.ndarray,
                  token_ids: np.ndarray) -> float:
    obs = np.exp(_log_softmax(_validate_logits(observer_logits)))
    perf = np.exp(_log_softmax(_validate_logits(performer_logits)))
    if obs.shape != perf.shape:
        raise ValueError("logit arrays must have the same shape")
    ids = np.asarray(token_ids, dtype=int)
    delta = np.abs(_ranks(obs, ids) - _ranks(perf, ids))
    return float(np.var(delta))


def feature_vector(observer_logits: np.ndarray, performer_logits: np.ndarray,
                   token_ids: np.ndarray) -> dict[str, float]:
    """Compute the retained feature set from one aligned model pass."""
    return {
        "binoculars_score": binoculars_score(observer_logits, performer_logits, token_ids),
        "rank_volatility": rank_volatility(performer_logits, token_ids),
        "stc": surprisal_trajectory_curvature(performer_logits, token_ids),
        "cmrd_var": cmrd_variance(observer_logits, performer_logits, token_ids),
        "roughness": roughness(performer_logits, token_ids),
        "tar": token_agreement_rate(observer_logits, performer_logits),
    }
