from __future__ import annotations

import numpy as np
import pytest

from occupancy_ratio_benchmark.d4rl_ope import (
    _D4RLTransitions,
    _load_d4rl_dataset_transitions,
    _load_d4rl_dataset_transitions_cached,
    _partition_d4rl_transitions,
)


def _transitions() -> _D4RLTransitions:
    episode_ids = np.repeat(np.arange(10), 3)
    rows = episode_ids.size
    return _D4RLTransitions(
        states=np.arange(rows * 2, dtype=np.float64).reshape(rows, 2),
        actions=np.arange(rows, dtype=np.float64).reshape(rows, 1),
        next_states=np.arange(rows * 2, dtype=np.float64).reshape(rows, 2) + 1.0,
        rewards=np.arange(rows, dtype=np.float64),
        masks=np.ones(rows, dtype=np.float64),
        episode_ids=episode_ids,
        timesteps=np.tile(np.arange(3), 10),
        initial_states=np.arange(20, dtype=np.float64).reshape(10, 2),
        initial_episode_ids=np.arange(10),
    )


def test_d4rl_raw_episode_partition_is_disjoint_and_reproducible() -> None:
    kwargs = {
        "audit_fraction": 0.2,
        "split_seed": 37,
        "split_key": "hopper-medium-v0",
    }
    train, train_metadata = _partition_d4rl_transitions(
        _transitions(), partition="train", **kwargs
    )
    audit, audit_metadata = _partition_d4rl_transitions(
        _transitions(), partition="audit", **kwargs
    )
    repeated, repeated_metadata = _partition_d4rl_transitions(
        _transitions(), partition="audit", **kwargs
    )

    train_ids = set(train.initial_episode_ids.tolist())
    audit_ids = set(audit.initial_episode_ids.tolist())
    assert train_ids.isdisjoint(audit_ids)
    assert train_ids | audit_ids == set(range(10))
    assert len(train_ids) == 8
    assert len(audit_ids) == 2
    assert set(train.episode_ids.tolist()) == train_ids
    assert set(audit.episode_ids.tolist()) == audit_ids
    assert train_metadata["behavior_episode_partition_sha256"] == audit_metadata[
        "behavior_episode_partition_sha256"
    ]
    assert repeated_metadata == audit_metadata
    np.testing.assert_array_equal(repeated.episode_ids, audit.episode_ids)


def test_d4rl_partition_changes_with_split_key() -> None:
    first, _ = _partition_d4rl_transitions(
        _transitions(),
        partition="audit",
        audit_fraction=0.3,
        split_seed=11,
        split_key="hopper-medium-v0",
    )
    second, _ = _partition_d4rl_transitions(
        _transitions(),
        partition="audit",
        audit_fraction=0.3,
        split_seed=11,
        split_key="walker2d-medium-v0",
    )

    assert not np.array_equal(first.initial_episode_ids, second.initial_episode_ids)


def test_d4rl_loader_reuses_immutable_transition_table(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "dataset.hdf5"
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.arange(12, dtype=np.float64).reshape(4, 3)
        handle["next_observations"] = np.arange(12, dtype=np.float64).reshape(4, 3) + 1
        handle["actions"] = np.arange(8, dtype=np.float64).reshape(4, 2)
        handle["rewards"] = np.arange(4, dtype=np.float64)
        handle["terminals"] = np.asarray([0, 1, 0, 1], dtype=np.float64)
        handle["timeouts"] = np.zeros(4, dtype=np.float64)

    _load_d4rl_dataset_transitions_cached.cache_clear()
    first = _load_d4rl_dataset_transitions(path)
    second = _load_d4rl_dataset_transitions(path)

    assert first is second
    assert _load_d4rl_dataset_transitions_cached.cache_info().hits == 1
    _load_d4rl_dataset_transitions_cached.cache_clear()
