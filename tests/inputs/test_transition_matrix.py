"""Unit tests for TransitionMatrix."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.inputs.transition_matrix import TransitionMatrix

FROM_STATES = ["A", "B"]
TO_STATES = ["A", "B", "D"]


def make_valid_prob_df():
    return pd.DataFrame(
        [[0.90, 0.08, 0.02], [0.05, 0.90, 0.05]],
        index=FROM_STATES,
        columns=TO_STATES,
    )


def make_valid(**overrides):
    kwargs = dict(probabilities=make_valid_prob_df())
    kwargs.update(overrides)
    return TransitionMatrix(**kwargs)


class TestConstructionFromDataFrame:
    def test_states_default_to_index_and_columns(self):
        tm = make_valid()
        assert tm.from_states == FROM_STATES
        assert tm.to_states == TO_STATES

    def test_matching_states_accepted(self):
        tm = make_valid(from_states=FROM_STATES, to_states=TO_STATES)
        assert tm.from_states == FROM_STATES
        assert tm.to_states == TO_STATES

    def test_mismatched_from_states_raises(self):
        with pytest.raises(ValueError, match="does not match probabilities index"):
            make_valid(from_states=["x", "y"])

    def test_mismatched_to_states_raises(self):
        with pytest.raises(ValueError, match="does not match probabilities columns"):
            make_valid(to_states=["x", "y", "D"])


class TestConstructionFromNdarray:
    def test_requires_states(self):
        with pytest.raises(ValueError, match="must be provided when probabilities is a raw ndarray"):
            TransitionMatrix(probabilities=make_valid_prob_df().to_numpy())

    def test_valid_ndarray_accepted(self):
        tm = TransitionMatrix(
            probabilities=make_valid_prob_df().to_numpy(),
            from_states=FROM_STATES,
            to_states=TO_STATES,
        )
        assert tm.from_states == FROM_STATES
        assert tm.n_from_states == 2
        assert tm.n_to_states == 3


class TestValidation:
    def test_non_2d_probabilities_raises(self):
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            TransitionMatrix(
                probabilities=np.array([0.5, 0.5]),
                from_states=["A"],
                to_states=["A", "D"],
            )

    def test_shape_state_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match number of"):
            TransitionMatrix(
                probabilities=make_valid_prob_df().to_numpy(),
                from_states=["A", "B", "C"],
                to_states=TO_STATES,
            )

    def test_duplicate_from_states_raises(self):
        prob = make_valid_prob_df()
        prob.index = ["A", "A"]
        with pytest.raises(ValueError, match="from_states must be unique"):
            TransitionMatrix(probabilities=prob)

    def test_duplicate_to_states_raises(self):
        prob = make_valid_prob_df()
        prob.columns = ["A", "A", "D"]
        with pytest.raises(ValueError, match="to_states must be unique"):
            TransitionMatrix(probabilities=prob)

    def test_default_state_not_last_raises(self):
        with pytest.raises(ValueError, match="must be the last entry"):
            make_valid(default_state="A")

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, np.nan])
    def test_probabilities_outside_unit_interval_raise(self, bad_value):
        prob = make_valid_prob_df()
        prob.iloc[0, 0] = bad_value
        with pytest.raises(ValueError, match="must be within"):
            make_valid(probabilities=prob)

    def test_row_not_summing_to_one_raises(self):
        prob = make_valid_prob_df()
        prob.iloc[1] = [0.5, 0.4, 0.05]
        with pytest.raises(ValueError, match=r"must sum to 1: \['B'\]"):
            make_valid(probabilities=prob)

    def test_to_state_without_matching_from_state_raises(self):
        prob = pd.DataFrame(
            [[0.90, 0.08, 0.02], [0.05, 0.90, 0.05]],
            index=["A", "B"],
            columns=["A", "C", "D"],
        )
        with pytest.raises(ValueError, match="have no matching from-state row"):
            TransitionMatrix(probabilities=prob)


class TestDerivedAttributes:
    def test_state_index_maps_names_to_rows(self):
        tm = make_valid()
        assert tm.state_index == {"A": 0, "B": 1}

    def test_thresholds_match_reversed_cumulative_probabilities(self):
        tm = make_valid()
        prob = make_valid_prob_df().to_numpy()
        expected = norm.ppf(np.cumsum(prob[:, ::-1], axis=1)[:, :-1])
        assert tm.thresholds.shape == (2, 2)
        np.testing.assert_allclose(tm.thresholds, expected)

    def test_thresholds_rows_are_non_decreasing(self):
        tm = make_valid()
        assert (np.diff(tm.thresholds, axis=1) >= 0).all()

    def test_default_probs_are_last_column(self):
        tm = make_valid()
        np.testing.assert_array_equal(tm.default_probs, [0.02, 0.05])

    def test_default_probs_worst_first_matches_own_state_default_probs(self):
        # to_states (excluding D) worst-first is ["B", "A"]; each state's own
        # default probability is its from-state row's D-column value.
        tm = make_valid()
        np.testing.assert_array_equal(tm.default_probs_worst_first, [0.05, 0.02])

    def test_state_counts(self):
        tm = make_valid()
        assert tm.n_from_states == 2
        assert tm.n_to_states == 3
