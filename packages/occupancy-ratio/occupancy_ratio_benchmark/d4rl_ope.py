from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.occupancy_sampling import (
    append_nonabsorbing_indicator,
    sample_absorbing_discounted_occupancy,
)
from occupancy_ratio_benchmark.tabular import OptionalDatasetUnavailable


Array = np.ndarray


D4RL_OPE_POLICY_TABLE: dict[str, dict[str, Any]] = {
    "cheetah-random": dict(env_id="halfcheetah-random-v0", stem="cheetah_random_params", return_undiscounted=-199.0),
    "cheetah-medium": dict(env_id="halfcheetah-medium-v0", stem="cheetah_medium_params", return_undiscounted=3985.0),
    "cheetah-medium-high": dict(
        env_id="halfcheetah-medium-v0", stem="cheetah_medium_high_params", return_undiscounted=6751.0
    ),
    "cheetah": dict(env_id="halfcheetah-expert-v0", stem="cheetah_params", return_undiscounted=12330.0),
    "hopper-random": dict(env_id="hopper-random-v0", stem="hopper_random_params", return_undiscounted=1257.0),
    "hopper-medium": dict(env_id="hopper-medium-v0", stem="hopper_medium_params", return_undiscounted=2260.0),
    "hopper-medium-high": dict(
        env_id="hopper-medium-v0", stem="hopper_medium_high_params", return_undiscounted=3256.0
    ),
    "hopper": dict(env_id="hopper-expert-v0", stem="hopper_params", return_undiscounted=3624.0),
    "walker2d-random": dict(env_id="walker2d-random-v0", stem="walker2d_random_params", return_undiscounted=896.0),
    "walker2d-medium-low": dict(
        env_id="walker2d-medium-v0", stem="walker2d_medium_low_params", return_undiscounted=1555.0
    ),
    "walker2d-medium": dict(env_id="walker2d-medium-v0", stem="walker2d_medium_params", return_undiscounted=2760.0),
    "walker2d": dict(env_id="walker2d-expert-v0", stem="walker2d_params", return_undiscounted=4005.0),
}

_D4RL_GYM_MUJOCO_DATASET_URLS: dict[str, str] = {
    "halfcheetah-random-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5",
    "halfcheetah-medium-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5",
    "halfcheetah-expert-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5",
    "hopper-random-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5",
    "hopper-medium-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5",
    "hopper-expert-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5",
    "walker2d-random-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5",
    "walker2d-medium-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5",
    "walker2d-expert-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5",
}

_ONNX_BASE_URLS = (
    "https://rail.eecs.berkeley.edu/datasets/offline_rl/ope_policies/onnx",
    "http://rail.eecs.berkeley.edu/datasets/offline_rl/ope_policies/onnx",
)


@dataclass(frozen=True)
class _D4RLTransitions:
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    masks: Array
    episode_ids: Array
    timesteps: Array
    initial_states: Array
    initial_episode_ids: Array


@dataclass
class D4RLOPEPolicy:
    sampler_path: Path
    log_prob_path: Path

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            raise OptionalDatasetUnavailable("Install onnxruntime to use D4RL OPE ONNX policies.") from exc
        self._session = ort.InferenceSession(str(self.sampler_path), providers=["CPUExecutionProvider"])
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    def sample(self, observations: Array, rng: np.random.Generator, *, action_dim: int) -> Array:
        obs = np.asarray(observations, dtype=np.float32).reshape(np.asarray(observations).shape[0], -1)
        noise = rng.normal(size=(obs.shape[0], int(action_dim))).astype(np.float32)
        feeds: dict[str, Array] = {}
        for name in self._input_names:
            lower = str(name).lower()
            feeds[name] = noise if "noise" in lower or "eps" in lower or "normal" in lower else obs
        if len(feeds) == 1:
            feeds[self._input_names[0]] = obs
        outputs = self._session.run(None, feeds)
        action = outputs[0]
        return np.asarray(action, dtype=np.float64).reshape(obs.shape[0], int(action_dim))


def make_d4rl_ope_dataset(
    *,
    policy_id: str,
    gamma: float,
    sample_size: int,
    seed: int,
    asset_cache_dir: str | Path,
    install_assets: bool,
    target_value_rollouts: int,
    target_occupancy_trajectories_per_pool: int = 0,
    require_exact_rollout_env: bool = True,
    behavior_episode_partition: str = "all",
    behavior_audit_fraction: float = 0.2,
    behavior_partition_seed: int = 0,
    include_target_evaluation: bool = True,
) -> BenchmarkDataset:
    """Create a standard D4RL MuJoCo OPE dataset with real D4RL policy artifacts."""

    if policy_id not in D4RL_OPE_POLICY_TABLE:
        raise ValueError(f"Unknown D4RL OPE policy id '{policy_id}'.")
    spec = D4RL_OPE_POLICY_TABLE[str(policy_id)]
    asset_root = Path(asset_cache_dir)
    policy_paths = _ensure_onnx_policy_assets(
        policy_id=str(policy_id),
        stem=str(spec["stem"]),
        asset_cache_dir=asset_root,
        install_assets=bool(install_assets),
    )
    policy = D4RLOPEPolicy(*policy_paths)
    env_id = str(spec["env_id"])
    dataset_path = _ensure_d4rl_dataset_asset(
        env_id=env_id,
        asset_cache_dir=asset_root,
        install_assets=bool(install_assets),
    )
    all_transitions = _load_d4rl_dataset_transitions(dataset_path)
    transitions, partition_metadata = _partition_d4rl_transitions(
        all_transitions,
        partition=str(behavior_episode_partition),
        audit_fraction=float(behavior_audit_fraction),
        split_seed=int(behavior_partition_seed),
        split_key=env_id,
    )
    rng = np.random.default_rng(int(seed) + 53_001)
    n = transitions.states.shape[0]
    if n <= 0:
        raise OptionalDatasetUnavailable(f"D4RL dataset '{env_id}' returned no usable transitions.")
    action_dim = int(np.asarray(transitions.actions).reshape(n, -1).shape[1])
    occupancy = sample_absorbing_discounted_occupancy(
        states=transitions.states,
        actions=transitions.actions,
        next_states=transitions.next_states,
        rewards=transitions.rewards,
        episode_ids=transitions.episode_ids,
        timesteps=transitions.timesteps,
        gamma=float(gamma),
        sample_size=int(sample_size),
        rng=rng,
    )
    target_actions = np.zeros((int(sample_size), action_dim), dtype=np.float64)
    nonabsorbing = ~occupancy.is_absorbing
    target_actions[nonabsorbing] = policy.sample(
        occupancy.states[nonabsorbing, :-1], rng, action_dim=action_dim
    )
    next_target_actions = np.zeros((int(sample_size), action_dim), dtype=np.float64)
    next_nonabsorbing = ~occupancy.next_is_absorbing
    next_target_actions[next_nonabsorbing] = policy.sample(
        occupancy.next_states[next_nonabsorbing, :-1], rng, action_dim=action_dim
    )
    initial_actions = policy.sample(transitions.initial_states, rng, action_dim=action_dim)
    if include_target_evaluation:
        target_value, target_value_se, target_value_status = _estimate_or_load_d4rl_target_value(
            policy_id=str(policy_id),
            asset_cache_dir=asset_root,
            env=None,
            policy=policy,
            gamma=float(gamma),
            rollouts=int(target_value_rollouts),
            action_dim=action_dim,
        )
        (
            target_occ_states,
            target_occ_actions,
            target_occ_episode_ids,
            target_occ_pool_ids,
            target_occ_status,
        ) = _estimate_or_load_d4rl_target_occupancy(
            policy_id=str(policy_id),
            asset_cache_dir=asset_root,
            env=None,
            policy=policy,
            gamma=float(gamma),
            trajectories_per_pool=int(target_occupancy_trajectories_per_pool),
            state_dim=int(transitions.states.shape[1]),
            action_dim=action_dim,
        )
    else:
        target_value = target_value_se = float("nan")
        target_value_status = "independent_behavior_audit_disabled"
        target_occ_states = target_occ_actions = None
        target_occ_episode_ids = target_occ_pool_ids = None
        target_occ_status = "disabled"
    needs_value = bool(include_target_evaluation) and not np.isfinite(target_value)
    needs_occupancy = (
        bool(include_target_evaluation)
        and int(target_occupancy_trajectories_per_pool) > 0
        and target_occ_states is None
    )
    rollout_env = None
    rollout_env_status = "cache_complete"
    if needs_value or needs_occupancy:
        rollout_env, rollout_env_status = _make_d4rl_rollout_env(env_id)
    if rollout_env is not None:
        target_value, target_value_se, target_value_status = _estimate_or_load_d4rl_target_value(
            policy_id=str(policy_id),
            asset_cache_dir=asset_root,
            env=rollout_env,
            policy=policy,
            gamma=float(gamma),
            rollouts=int(target_value_rollouts),
            action_dim=action_dim,
        )
        (
            target_occ_states,
            target_occ_actions,
            target_occ_episode_ids,
            target_occ_pool_ids,
            target_occ_status,
        ) = _estimate_or_load_d4rl_target_occupancy(
            policy_id=str(policy_id),
            asset_cache_dir=asset_root,
            env=rollout_env,
            policy=policy,
            gamma=float(gamma),
            seed=_stable_d4rl_target_seed(
                f"{policy_id}-occupancy",
                gamma=float(gamma),
                rollouts=2 * int(target_occupancy_trajectories_per_pool),
            ),
            trajectories_per_pool=int(target_occupancy_trajectories_per_pool),
            state_dim=int(transitions.states.shape[1]),
            action_dim=action_dim,
        )
        try:
            rollout_env.close()
        except Exception:
            pass
    if bool(include_target_evaluation) and bool(require_exact_rollout_env) and (
        not np.isfinite(target_value)
        or (int(target_occupancy_trajectories_per_pool) > 0 and target_occ_states is None)
    ):
        raise OptionalDatasetUnavailable(
            "D4RL confirmatory evaluation requires exact Gym v0 target caches; "
            f"preflight failed with {rollout_env_status}, value={target_value_status}, "
            f"occupancy={target_occ_status}. Run the legacy v0 cache generator first."
        )

    metadata = {
        "benchmark_track": "d4rl_ope",
        "dataset_variant": str(policy_id),
        "d4rl_policy_id": str(policy_id),
        "d4rl_env_id": env_id,
        "d4rl_dataset_path": str(dataset_path),
        "d4rl_policy_sampler_path": str(policy.sampler_path),
        "d4rl_policy_log_prob_path": str(policy.log_prob_path),
        "d4rl_reference_return_undiscounted": float(spec["return_undiscounted"]),
        "d4rl_reference_return_source": "D4RL OPE wiki undiscounted_10_seed_average_metadata_only",
        "target_policy_value": target_value if np.isfinite(target_value) else np.nan,
        "target_policy_value_se": target_value_se if np.isfinite(target_value_se) else np.nan,
        "target_policy_value_kind": "normalized_discounted_step_reward_mc_rollout",
        "target_policy_value_source": "d4rl_gym_v0_discounted_mc_rollout",
        "target_value_rollouts": int(target_value_rollouts),
        "target_value_status": target_value_status,
        "target_occupancy_trajectories_per_pool": int(target_occupancy_trajectories_per_pool),
        "target_occupancy_status": target_occ_status,
        "truth_source": "target_policy_mc_rollout" if np.isfinite(target_value) else "unavailable",
        "reference_distribution": "normalized_discounted_behavior_occupancy_sampled",
        "reference_sampling": "uniform_episode_geometric_time_with_absorbing_tail",
        "reference_sampling_gamma": float(gamma),
        "reference_sampling_replacement": 1.0,
        "rollout_env_backend": "d4rl_gym_v0",
        "rollout_env_status": rollout_env_status,
        "compatibility_note": (
            "D4RL HDF5 logged data, target policy, and target rollouts must share the exact "
            "D4RL Gym v0 MDP; Gymnasium v4 rollouts are rejected."
        ),
        "fold_grouping": "episode",
        "state_dim": int(occupancy.states.shape[1]),
        "action_dim": int(action_dim),
        "has_ratio_truth": 0.0,
        "absorbing_state_contract": 1.0,
        "absorbing_source_fraction": float(np.mean(occupancy.is_absorbing)),
        **partition_metadata,
    }
    return BenchmarkDataset(
        setting="d4rl_ope",
        states=occupancy.states,
        actions=occupancy.actions,
        next_states=occupancy.next_states,
        target_actions=target_actions,
        next_target_actions=next_target_actions,
        rewards=occupancy.rewards,
        true_ratio=None,
        true_action_ratio=None,
        true_transition_ratio=None,
        initial_states=append_nonabsorbing_indicator(transitions.initial_states),
        initial_actions=initial_actions,
        initial_weights=np.ones(transitions.initial_states.shape[0], dtype=np.float64),
        masks=occupancy.masks,
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        episode_ids=np.asarray(occupancy.episode_ids, dtype=np.int64),
        timesteps=occupancy.timesteps,
        initial_episode_ids=transitions.initial_episode_ids.astype(np.int64, copy=False),
        is_absorbing=occupancy.is_absorbing,
        target_occupancy_states=target_occ_states,
        target_occupancy_actions=target_occ_actions,
        target_occupancy_episode_ids=target_occ_episode_ids,
        target_occupancy_pool_ids=target_occ_pool_ids,
        target_policy_value=target_value if np.isfinite(target_value) else None,
        target_policy_value_se=target_value_se if np.isfinite(target_value_se) else None,
        target_policy_value_kind="normalized_discounted_step_reward_mc_rollout",
        metadata=metadata,
    )


def _discounted_transition_sampling_probs(timesteps: Array, *, gamma: float) -> Array:
    times = np.asarray(timesteps, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise OptionalDatasetUnavailable("D4RL transition table is empty.")
    weights = np.power(float(gamma), times)
    weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise OptionalDatasetUnavailable("D4RL discounted transition sampling weights are all zero.")
    return (weights / total).astype(np.float64, copy=False)


def _ensure_onnx_policy_assets(
    *,
    policy_id: str,
    stem: str,
    asset_cache_dir: Path,
    install_assets: bool,
) -> tuple[Path, Path]:
    policy_dir = asset_cache_dir / "d4rl_ope" / "onnx"
    sampler = policy_dir / f"{stem}.sampler.onnx"
    log_prob = policy_dir / f"{stem}.log_prob.onnx"
    missing = [path for path in (sampler, log_prob) if not path.exists()]
    if missing and install_assets:
        policy_dir.mkdir(parents=True, exist_ok=True)
        for path in missing:
            last_error = ""
            for base_url in _ONNX_BASE_URLS:
                url = f"{base_url}/{path.name}"
                try:
                    urllib.request.urlretrieve(url, path)  # noqa: S310 - explicit benchmark asset URL.
                    last_error = ""
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
            if last_error:
                raise OptionalDatasetUnavailable(
                    f"D4RL OPE policy asset download failed for {policy_id} ({path.name}): {last_error}"
                )
    if not sampler.exists() or not log_prob.exists():
        raise OptionalDatasetUnavailable(
            f"Missing D4RL OPE policy ONNX assets for {policy_id} under {policy_dir}; "
            "rerun with install_benchmark_assets=True."
        )
    return sampler, log_prob


def _ensure_d4rl_dataset_asset(*, env_id: str, asset_cache_dir: Path, install_assets: bool) -> Path:
    if env_id not in _D4RL_GYM_MUJOCO_DATASET_URLS:
        raise OptionalDatasetUnavailable(f"No direct D4RL HDF5 URL is configured for '{env_id}'.")
    dataset_dir = asset_cache_dir / "d4rl_ope" / "datasets"
    dataset_path = dataset_dir / f"{env_id}.hdf5"
    if dataset_path.exists():
        return dataset_path
    if not install_assets:
        raise OptionalDatasetUnavailable(
            f"Missing D4RL dataset asset for {env_id} under {dataset_dir}; rerun with install_benchmark_assets=True."
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    url = _D4RL_GYM_MUJOCO_DATASET_URLS[env_id]
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 - explicit D4RL benchmark asset URL.
        tmp_path.replace(dataset_path)
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise OptionalDatasetUnavailable(
            f"D4RL dataset download failed for {env_id}: {type(exc).__name__}: {exc}"
        ) from exc
    return dataset_path


def _load_d4rl_dataset_transitions(dataset_path: Path) -> _D4RLTransitions:
    try:
        import h5py  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        raise OptionalDatasetUnavailable("Install h5py to load direct D4RL HDF5 datasets.") from exc
    try:
        with h5py.File(dataset_path, "r") as handle:
            raw = {key: np.asarray(handle[key]) for key in handle.keys()}
    except Exception as exc:
        raise OptionalDatasetUnavailable(
            f"Could not read D4RL dataset {dataset_path}: {type(exc).__name__}: {exc}"
        ) from exc
    return _transitions_from_d4rl_raw(raw)


def _partition_d4rl_transitions(
    transitions: _D4RLTransitions,
    *,
    partition: str,
    audit_fraction: float,
    split_seed: int,
    split_key: str,
) -> tuple[_D4RLTransitions, dict[str, Any]]:
    """Apply a deterministic raw-episode train/audit partition."""

    choice = str(partition)
    if choice not in {"all", "train", "audit"}:
        raise ValueError("behavior_episode_partition must be 'all', 'train', or 'audit'")
    fraction = float(audit_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("behavior_audit_fraction must lie strictly between zero and one")
    episode_ids = np.unique(np.asarray(transitions.initial_episode_ids, dtype=np.int64))
    if episode_ids.size < 3:
        raise OptionalDatasetUnavailable("D4RL behavior audit requires at least three episodes")
    token = hashlib.sha256(
        f"d4rl-behavior-audit|{split_key}|{int(split_seed)}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
    ordered = episode_ids[rng.permutation(episode_ids.size)]
    audit_count = min(episode_ids.size - 2, max(1, int(np.ceil(fraction * episode_ids.size))))
    audit_ids = np.sort(ordered[:audit_count])
    train_ids = np.sort(ordered[audit_count:])
    selected_ids = episode_ids if choice == "all" else (train_ids if choice == "train" else audit_ids)
    transition_mask = np.isin(transitions.episode_ids, selected_ids)
    initial_mask = np.isin(transitions.initial_episode_ids, selected_ids)
    if not np.any(transition_mask) or not np.any(initial_mask):
        raise OptionalDatasetUnavailable(f"D4RL {choice} episode partition is empty")
    selected = _D4RLTransitions(
        states=np.asarray(transitions.states)[transition_mask],
        actions=np.asarray(transitions.actions)[transition_mask],
        next_states=np.asarray(transitions.next_states)[transition_mask],
        rewards=np.asarray(transitions.rewards)[transition_mask],
        masks=np.asarray(transitions.masks)[transition_mask],
        episode_ids=np.asarray(transitions.episode_ids)[transition_mask],
        timesteps=np.asarray(transitions.timesteps)[transition_mask],
        initial_states=np.asarray(transitions.initial_states)[initial_mask],
        initial_episode_ids=np.asarray(transitions.initial_episode_ids)[initial_mask],
    )
    split_digest = hashlib.sha256(
        np.asarray(audit_ids, dtype="<i8").tobytes()
        + np.asarray(train_ids, dtype="<i8").tobytes()
    ).hexdigest()
    return selected, {
        "behavior_episode_partition": choice,
        "behavior_episode_partition_seed": int(split_seed),
        "behavior_episode_audit_fraction": fraction,
        "behavior_episode_total_count": int(episode_ids.size),
        "behavior_episode_train_count": int(train_ids.size),
        "behavior_episode_audit_count": int(audit_ids.size),
        "behavior_episode_selected_count": int(selected_ids.size),
        "behavior_episode_partition_sha256": split_digest,
    }


def _extract_d4rl_transitions(env: Any) -> _D4RLTransitions:
    return _transitions_from_d4rl_raw(env.get_dataset())


def _transitions_from_d4rl_raw(raw: dict[str, Array]) -> _D4RLTransitions:
    observations = np.asarray(raw["observations"], dtype=np.float64)
    actions = np.asarray(raw["actions"], dtype=np.float64)
    rewards = np.asarray(raw["rewards"], dtype=np.float64).reshape(-1)
    terminals = np.asarray(raw.get("terminals", np.zeros(actions.shape[0])), dtype=bool).reshape(-1)
    timeouts = np.asarray(raw.get("timeouts", np.zeros(actions.shape[0])), dtype=bool).reshape(-1)
    has_next = "next_observations" in raw
    next_observations = np.asarray(raw["next_observations"], dtype=np.float64) if has_next else None
    states: list[Array] = []
    acts: list[Array] = []
    next_states: list[Array] = []
    rews: list[float] = []
    masks: list[float] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []
    initial_states: list[Array] = []
    initial_episode_ids: list[int] = []
    episode_id = 0
    timestep = 0
    new_episode = True
    n = int(actions.shape[0])
    for idx in range(n):
        if new_episode:
            initial_states.append(observations[idx].copy())
            initial_episode_ids.append(int(episode_id))
            timestep = 0
        done = bool(terminals[idx] or timeouts[idx])
        if has_next:
            next_obs = next_observations[idx]
            keep = True
        else:
            keep = idx + 1 < n and not (bool(timeouts[idx]) and not bool(terminals[idx]))
            next_obs = observations[idx + 1] if idx + 1 < n else observations[idx]
        if keep:
            states.append(observations[idx].copy())
            acts.append(actions[idx].copy())
            next_states.append(np.asarray(next_obs).copy())
            rews.append(float(rewards[idx]))
            masks.append(float(not bool(terminals[idx])))
            episode_ids.append(int(episode_id))
            timesteps.append(int(timestep))
        timestep += 1
        if done:
            episode_id += 1
            new_episode = True
        else:
            new_episode = False
    return _D4RLTransitions(
        states=np.asarray(states, dtype=np.float64).reshape(len(states), -1),
        actions=np.asarray(acts, dtype=np.float64).reshape(len(acts), -1),
        next_states=np.asarray(next_states, dtype=np.float64).reshape(len(next_states), -1),
        rewards=np.asarray(rews, dtype=np.float64),
        masks=np.asarray(masks, dtype=np.float64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        initial_states=np.asarray(initial_states, dtype=np.float64).reshape(len(initial_states), -1),
        initial_episode_ids=np.asarray(initial_episode_ids, dtype=np.int64),
    )


def _make_d4rl_rollout_env(env_id: str) -> tuple[Any | None, str]:
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    try:
        import gym  # noqa: PLC0415
        import d4rl  # noqa: F401, PLC0415
        import d4rl.gym_mujoco  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, f"d4rl_gym_v0_import_failed:{type(exc).__name__}:{exc}"
    try:
        return gym.make(str(env_id)), f"d4rl_gym_v0:{env_id}"
    except Exception as exc:
        return None, f"d4rl_gym_v0_make_failed:{env_id}:{type(exc).__name__}:{exc}"


def _estimate_d4rl_target_value(
    *,
    env: Any,
    policy: D4RLOPEPolicy,
    gamma: float,
    seed: int,
    rollouts: int,
    action_dim: int,
) -> tuple[float, float, str]:
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    max_steps = int(getattr(getattr(env, "spec", None), "max_episode_steps", None) or 1_000)
    try:
        _seed_env(env, int(seed))
        for _ in range(int(rollouts)):
            obs = _reset_env(env, rng)
            total = 0.0
            discount = 1.0
            for _t in range(max_steps):
                action = policy.sample(obs.reshape(1, -1), rng, action_dim=int(action_dim)).reshape(-1)
                obs, reward, done = _step_env(env, action)
                total += discount * float(reward)
                discount *= float(gamma)
                if done:
                    break
            values.append(float((1.0 - float(gamma)) * total))
    except Exception as exc:
        return float("nan"), float("nan"), f"mc_rollout_failed:{type(exc).__name__}:{exc}"
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), "mc_rollout_empty"
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return float(np.mean(arr)), se, "mc_rollout_ok"


def _estimate_d4rl_target_occupancy(
    *,
    env: Any,
    policy: D4RLOPEPolicy,
    gamma: float,
    seed: int,
    trajectories_per_pool: int,
    state_dim: int,
    action_dim: int,
) -> tuple[Array | None, Array | None, Array | None, Array | None, str]:
    per_pool = int(trajectories_per_pool)
    if per_pool <= 0:
        return None, None, None, None, "disabled"
    rng = np.random.default_rng(int(seed))
    total = 2 * per_pool
    states = np.zeros((total, int(state_dim) + 1), dtype=np.float64)
    actions = np.zeros((total, int(action_dim)), dtype=np.float64)
    try:
        _seed_env(env, int(seed))
        for trajectory in range(total):
            target_time = 0 if float(gamma) == 0.0 else int(rng.geometric(1.0 - float(gamma)) - 1)
            obs = _reset_env(env, rng)
            absorbed = False
            for _ in range(target_time):
                action = policy.sample(obs.reshape(1, -1), rng, action_dim=int(action_dim)).reshape(-1)
                obs, _reward, done = _step_env(env, action)
                if done:
                    absorbed = True
                    break
            if absorbed:
                states[trajectory, -1] = 1.0
            else:
                states[trajectory, :-1] = obs
                actions[trajectory] = policy.sample(
                    obs.reshape(1, -1), rng, action_dim=int(action_dim)
                ).reshape(-1)
    except Exception as exc:
        return None, None, None, None, f"mc_rollout_failed:{type(exc).__name__}:{exc}"
    return (
        states,
        actions,
        np.arange(total, dtype=np.int64),
        np.repeat(np.arange(2, dtype=np.int64), per_pool),
        "mc_rollout_ok",
    )


def _estimate_or_load_d4rl_target_value(
    *,
    policy_id: str,
    asset_cache_dir: Path,
    env: Any | None,
    policy: D4RLOPEPolicy,
    gamma: float,
    rollouts: int,
    action_dim: int,
) -> tuple[float, float, str]:
    cache_path = _d4rl_target_value_cache_path(
        asset_cache_dir=asset_cache_dir,
        policy_id=str(policy_id),
        gamma=float(gamma),
        rollouts=int(rollouts),
    )
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_se = payload.get("target_value_se")
            return (
                float(payload["target_value"]),
                float(cached_se) if cached_se is not None else float("nan"),
                f"cache:{payload.get('status', 'mc_rollout_ok')}",
            )
        except Exception:
            pass
    if env is None:
        return float("nan"), float("nan"), "cache_missing"
    target_value, target_value_se, status = _estimate_d4rl_target_value(
        env=env,
        policy=policy,
        gamma=float(gamma),
        seed=_stable_d4rl_target_seed(str(policy_id), gamma=float(gamma), rollouts=int(rollouts)),
        rollouts=int(rollouts),
        action_dim=int(action_dim),
    )
    if np.isfinite(target_value):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "policy_id": str(policy_id),
                "gamma": float(gamma),
                "rollouts": int(rollouts),
                "target_value": float(target_value),
                "target_value_se": float(target_value_se) if np.isfinite(target_value_se) else None,
                "status": str(status),
            },
            sort_keys=True,
        )
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(cache_path)
    return target_value, target_value_se, status


def _estimate_or_load_d4rl_target_occupancy(
    *,
    policy_id: str,
    asset_cache_dir: Path,
    env: Any | None,
    policy: D4RLOPEPolicy,
    gamma: float,
    trajectories_per_pool: int,
    state_dim: int,
    action_dim: int,
    seed: int | None = None,
) -> tuple[Array | None, Array | None, Array | None, Array | None, str]:
    per_pool = int(trajectories_per_pool)
    if per_pool <= 0:
        return None, None, None, None, "disabled"
    cache_path = _d4rl_target_occupancy_cache_path(
        asset_cache_dir=asset_cache_dir,
        policy_id=str(policy_id),
        gamma=float(gamma),
        trajectories_per_pool=per_pool,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
    )
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                states = np.asarray(payload["states"], dtype=np.float64)
                actions = np.asarray(payload["actions"], dtype=np.float64)
                episode_ids = np.asarray(payload["episode_ids"], dtype=np.int64)
                pool_ids = np.asarray(payload["pool_ids"], dtype=np.int64)
            expected = 2 * per_pool
            if states.shape != (expected, int(state_dim) + 1):
                raise ValueError(f"unexpected cached state shape {states.shape}")
            if actions.shape != (expected, int(action_dim)):
                raise ValueError(f"unexpected cached action shape {actions.shape}")
            if episode_ids.shape != (expected,) or pool_ids.shape != (expected,):
                raise ValueError("unexpected cached id shape")
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError("cached target occupancy contains nonfinite values")
            return states, actions, episode_ids, pool_ids, "cache:mc_rollout_ok"
        except Exception:
            pass
    if env is None:
        return None, None, None, None, "cache_missing"
    rollout_seed = (
        int(seed)
        if seed is not None
        else _stable_d4rl_target_seed(
            f"{policy_id}-occupancy",
            gamma=float(gamma),
            rollouts=2 * per_pool,
        )
    )
    states, actions, episode_ids, pool_ids, status = _estimate_d4rl_target_occupancy(
        env=env,
        policy=policy,
        gamma=float(gamma),
        seed=rollout_seed,
        trajectories_per_pool=per_pool,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
    )
    if states is not None and actions is not None and episode_ids is not None and pool_ids is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                states=np.asarray(states, dtype=np.float64),
                actions=np.asarray(actions, dtype=np.float64),
                episode_ids=np.asarray(episode_ids, dtype=np.int64),
                pool_ids=np.asarray(pool_ids, dtype=np.int64),
            )
        tmp_path.replace(cache_path)
    return states, actions, episode_ids, pool_ids, status


def _d4rl_target_value_cache_path(
    *,
    asset_cache_dir: Path,
    policy_id: str,
    gamma: float,
    rollouts: int,
) -> Path:
    token = hashlib.sha1(
        f"d4rl_gym_v0|{policy_id}|{float(gamma):.12g}|{int(rollouts)}".encode("utf-8")
    ).hexdigest()[:12]
    safe_policy = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(policy_id))
    return asset_cache_dir / "d4rl_ope" / "target_values" / f"{safe_policy}_{token}.json"


def _d4rl_target_occupancy_cache_path(
    *,
    asset_cache_dir: Path,
    policy_id: str,
    gamma: float,
    trajectories_per_pool: int,
    state_dim: int,
    action_dim: int,
) -> Path:
    token = hashlib.sha1(
        (
            f"d4rl_gym_v0|{policy_id}|{float(gamma):.12g}|{int(trajectories_per_pool)}|"
            f"{int(state_dim)}|{int(action_dim)}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    safe_policy = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(policy_id))
    return asset_cache_dir / "d4rl_ope" / "target_occupancy" / f"{safe_policy}_{token}.npz"


def _stable_d4rl_target_seed(policy_id: str, *, gamma: float, rollouts: int) -> int:
    digest = hashlib.sha1(f"d4rl-target|{policy_id}|{float(gamma):.12g}|{int(rollouts)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _seed_env(env: Any, seed: int) -> None:
    try:
        env.seed(int(seed))
    except Exception:
        pass
    try:
        env.action_space.seed(int(seed) + 1)
    except Exception:
        pass
    try:
        env.observation_space.seed(int(seed) + 2)
    except Exception:
        pass


def _reset_env(env: Any, rng: np.random.Generator) -> Array:
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        out = env.reset(seed=seed)
    except TypeError:
        _seed_env(env, seed)
        out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _step_env(env: Any, action: Array) -> tuple[Array, float, bool]:
    act = np.asarray(action, dtype=np.float32)
    action_space = getattr(env, "action_space", None)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        act = np.clip(act, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
    step = env.step(act)
    if len(step) == 5:
        obs, reward, terminated, truncated, _info = step
        done = bool(terminated or truncated)
    else:
        obs, reward, done, _info = step
    return np.asarray(obs, dtype=np.float64).reshape(-1), float(reward), bool(done)
