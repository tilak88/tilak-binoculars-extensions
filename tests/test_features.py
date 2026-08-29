import numpy as np
import pytest

from binoculars_extension import (
    binoculars_score, cmrd_variance, feature_vector,
    rank_volatility, surprisal_trajectory_curvature,
)


def arrays():
    observer = np.array([[3., 1., 0., -1.], [0., 3., 1., -1.], [1., 0., 3., -1.], [3., 1., 0., -1.]])
    performer = np.array([[3., 1., 0., -1.], [0., 3., 1., -1.], [1., 0., 3., -1.], [3., 1., 0., -1.]])
    return observer, performer, np.array([0, 1, 2, 0])


def test_feature_vector_is_finite_and_complete():
    obs, perf, ids = arrays()
    result = feature_vector(obs, perf, ids)
    assert set(result) == {"binoculars_score", "rank_volatility", "stc", "cmrd_var", "roughness", "tar"}
    assert all(np.isfinite(v) for v in result.values())


def test_short_sequences_are_supported():
    obs, perf, ids = arrays()
    assert surprisal_trajectory_curvature(obs[:2], ids[:2]) == 0.0
    assert rank_volatility(perf[:1], ids[:1]) == 0.0


def test_shape_and_id_errors_are_explicit():
    obs, perf, ids = arrays()
    with pytest.raises(ValueError): binoculars_score(obs, perf[:2], ids)
    with pytest.raises(ValueError): cmrd_variance(obs, perf, np.array([9, 1, 2, 0]))

