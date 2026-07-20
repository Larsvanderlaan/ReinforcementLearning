from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio_benchmark.calibration_data import (
    DatasetPaths,
    build_calibration_dataset,
    calibration_group_ids,
    read_dataset_bundle,
    validate_normalized_dataset,
    write_dataset_bundle,
)
from occupancy_ratio_benchmark.discrete import make_discrete_dataset


def _bundle(tmp_path):
    return build_calibration_dataset(
        cell={
            "cell_id": "tabular",
            "benchmark_family": "random_tabular",
            "states": 8,
            "actions": 2,
            "policy_shift": 0.35,
        },
        axis_values={"sample_size": 128, "gamma": 0.9, "seed": 3},
        resolved_config={"truth": {"used_during_fit": False}},
        paths=DatasetPaths(asset_cache=tmp_path),
    )


def test_controlled_builder_and_lossless_round_trip(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    path = tmp_path / "dataset.json"
    metadata = write_dataset_bundle(path, bundle)
    restored = read_dataset_bundle(path)
    assert metadata["array_sha256"]
    assert restored.dataset.setting == "random_tabular_mdp"
    np.testing.assert_array_equal(restored.dataset.states, bundle.dataset.states)
    np.testing.assert_array_equal(restored.dataset.true_ratio, bundle.dataset.true_ratio)
    np.testing.assert_array_equal(restored.source_groups, bundle.source_groups)
    np.testing.assert_array_equal(restored.initial_groups, bundle.initial_groups)


def test_duplicate_transitions_share_group_but_initial_draws_do_not() -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=16,
        seed=2,
        n_states=4,
        n_actions=2,
    )
    dataset.states[1] = dataset.states[0]
    dataset.actions[1] = dataset.actions[0]
    dataset.next_states[1] = dataset.next_states[0]
    source, initial = calibration_group_ids(dataset)
    assert source[0] == source[1]
    assert np.unique(initial).size == initial.size


def test_stopped_dataset_is_rejected() -> None:
    dataset = make_discrete_dataset(
        setting="random_tabular_mdp",
        gamma=0.9,
        sample_size=16,
        seed=2,
    )
    dataset.initial_retention = np.zeros(dataset.initial_states.shape[0])
    with pytest.raises(ValueError, match="coverage-stopped"):
        validate_normalized_dataset(dataset)
