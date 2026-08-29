"""Numerical feature primitives for Binoculars-style detection."""

from .features import (
    binoculars_score,
    cross_entropy_sequence,
    feature_vector,
    cmrd_variance,
    rank_volatility,
    surprisal_trajectory_curvature,
    roughness,
    token_agreement_rate,
)

__all__ = [
    "binoculars_score", "cross_entropy_sequence", "feature_vector",
    "cmrd_variance", "rank_volatility", "surprisal_trajectory_curvature",
    "roughness", "token_agreement_rate",
]

