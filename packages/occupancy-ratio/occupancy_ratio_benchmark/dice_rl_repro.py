from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset, one_hot
from occupancy_ratio_benchmark.external_baselines import _ensure_dice_rl_importable
from occupancy_ratio_benchmark.occupancy_sampling import (
    append_nonabsorbing_indicator,
    sample_absorbing_discounted_occupancy,
)
from occupancy_ratio_benchmark.tabular import OptionalDatasetUnavailable


Array = np.ndarray


DICE_RL_REPRO_SETTINGS = {
    "dice_rl_cartpole": dict(env_name="cartpole", tabular_obs=False, policy_asset="testdata"),
    "dice_rl_cartpole_stopped_greedy_gate": dict(
        env_name="cartpole",
        tabular_obs=False,
        policy_asset="testdata",
    ),
    "dice_rl_reacher": dict(env_name="reacher", tabular_obs=False, policy_asset="testdata"),
    "dice_rl_taxi": dict(env_name="taxi", tabular_obs=True, policy_asset="taxi"),
}


@dataclass(frozen=True)
class _DiceRLEpisodes:
    states_raw: Array
    actions_raw: Array
    next_states_raw: Array
    rewards: Array
    masks: Array
    timesteps: Array
    episode_ids: Array
    initial_states_raw: Array
    initial_episode_ids: Array
    returns: Array
    action_spec: Any


def make_dice_rl_reproduction_dataset(
    *,
    setting: str,
    dataset_variant: str | None,
    gamma: float,
    sample_size: int,
    seed: int,
    dice_rl_repo_path: str | Path,
    asset_cache_dir: str | Path,
    install_assets: bool,
    num_trajectories: int,
    max_trajectory_length: int,
    target_value_rollouts: int,
    target_occupancy_trajectories_per_pool: int = 0,
    collection_batch_size: int = 20,
) -> BenchmarkDataset:
    """Create a DICE-RL original-style logged dataset from real policies."""

    if setting not in DICE_RL_REPRO_SETTINGS:
        raise ValueError(f"Unknown DICE-RL reproduction setting '{setting}'.")
    _patch_legacy_gym_seed()
    try:
        _ensure_dice_rl_importable(dice_rl_repo_path)
    except Exception as exc:
        raise OptionalDatasetUnavailable(f"DICE-RL import failed: {type(exc).__name__}: {exc}") from exc

    meta = DICE_RL_REPRO_SETTINGS[setting]
    env_name = str(meta["env_name"])
    tabular_obs = bool(meta["tabular_obs"])
    alpha = _parse_alpha_variant(dataset_variant)
    stopped_gate_track = setting == "dice_rl_cartpole_stopped_greedy_gate"
    behavior_alpha = 1.0 if stopped_gate_track else float(alpha)
    target_alpha = float(alpha) if stopped_gate_track else 1.0
    load_dir = _resolve_policy_load_dir(
        setting=setting,
        dice_rl_repo_path=Path(dice_rl_repo_path),
        asset_cache_dir=Path(asset_cache_dir),
        install_assets=bool(install_assets),
    )

    behavior = _collect_dice_rl_episodes(
        load_dir=load_dir,
        env_name=env_name,
        tabular_obs=tabular_obs,
        alpha=behavior_alpha,
        gamma=float(gamma),
        seed=int(seed),
        num_trajectories=int(num_trajectories),
        max_trajectory_length=int(max_trajectory_length),
        batch_size=int(collection_batch_size),
    )
    if behavior.states_raw.shape[0] == 0:
        raise OptionalDatasetUnavailable(f"No usable DICE-RL transitions collected for {setting}.")

    target_policy = _load_dice_rl_policy(
        load_dir=load_dir,
        env_name=env_name,
        tabular_obs=tabular_obs,
        alpha=target_alpha,
        seed=int(seed) + 101,
    )
    gate_policy = None
    if stopped_gate_track:
        gate_policy = _load_dice_rl_policy(
            load_dir=load_dir,
            env_name=env_name,
            tabular_obs=tabular_obs,
            alpha=behavior_alpha,
            seed=int(seed) + 211,
        )
        target_value = float("nan")
        target_value_se = float("nan")
        target_value_status = "pending_stopped_rollout"
    else:
        target_value, target_value_se, target_value_status = _estimate_or_load_dice_rl_target_value(
            setting=setting,
            load_dir=load_dir,
            env_name=env_name,
            tabular_obs=tabular_obs,
            gamma=float(gamma),
            seed=int(seed) + 91_001,
            target_value_rollouts=int(target_value_rollouts),
            max_trajectory_length=int(max_trajectory_length),
            collection_batch_size=int(collection_batch_size),
            asset_cache_dir=Path(asset_cache_dir),
        )

    action_encoder = _ActionEncoder.from_spec(behavior.action_spec)
    behavior_states = _flatten_features(behavior.states_raw)
    behavior_next_states = _flatten_features(behavior.next_states_raw)
    behavior_actions = action_encoder.encode(behavior.actions_raw)
    rng = np.random.default_rng(int(seed) + 37_001)
    occupancy = sample_absorbing_discounted_occupancy(
        states=behavior_states,
        actions=behavior_actions,
        next_states=behavior_next_states,
        rewards=behavior.rewards,
        episode_ids=behavior.episode_ids,
        timesteps=behavior.timesteps,
        gamma=float(gamma),
        sample_size=int(sample_size),
        rng=rng,
    )
    target_actions = np.zeros((int(sample_size), action_encoder.dim), dtype=np.float64)
    nonabsorbing = ~occupancy.is_absorbing
    source_idx = occupancy.source_indices[nonabsorbing]
    target_actions[nonabsorbing] = action_encoder.encode(
        _policy_actions(target_policy, behavior.states_raw[source_idx])
    )
    next_target_actions = np.zeros((int(sample_size), action_encoder.dim), dtype=np.float64)
    next_nonabsorbing = ~occupancy.next_is_absorbing
    next_idx = occupancy.source_indices[next_nonabsorbing]
    next_target_actions[next_nonabsorbing] = action_encoder.encode(
        _policy_actions(target_policy, behavior.next_states_raw[next_idx])
    )
    initial_states = append_nonabsorbing_indicator(_flatten_features(behavior.initial_states_raw))
    initial_actions = action_encoder.encode(_policy_actions(target_policy, behavior.initial_states_raw))
    next_retention = np.ones(int(sample_size), dtype=np.float64)
    initial_retention = np.ones(initial_states.shape[0], dtype=np.float64)
    if stopped_gate_track:
        if gate_policy is None:  # pragma: no cover - defensive
            raise RuntimeError("Stopped-gate CartPole requires a gate policy.")
        gate_next_actions = action_encoder.encode(
            _policy_mode_actions_batched(gate_policy, behavior.next_states_raw[next_idx])
        )
        next_retention[next_nonabsorbing] = _encoded_action_match(
            next_target_actions[next_nonabsorbing],
            gate_next_actions,
        )
        gate_initial_actions = action_encoder.encode(
            _policy_mode_actions_batched(gate_policy, behavior.initial_states_raw)
        )
        initial_retention = _encoded_action_match(initial_actions, gate_initial_actions)
    target_occ_states = None
    target_occ_actions = None
    target_occ_episode_ids = None
    target_occ_pool_ids = None
    target_occ_per_pool = int(target_occupancy_trajectories_per_pool)
    target_occupancy_mass = None
    stopped_mass_se = float("nan")
    target_evaluation_batch_size = int(collection_batch_size)
    actual_target_value_rollouts = int(target_value_rollouts)
    if stopped_gate_track:
        evaluation_trajectories = max(
            int(target_value_rollouts),
            2 * max(target_occ_per_pool, 1),
        )
        actual_target_value_rollouts = evaluation_trajectories
        target_evaluation_batch_size = max(int(collection_batch_size), 256)
        stopped_eval = _evaluate_infinite_cartpole_target_rollouts(
            target_policy=target_policy,
            gate_policy=gate_policy,
            action_encoder=action_encoder,
            gamma=float(gamma),
            seed=int(seed) + 191_001,
            num_trajectories=evaluation_trajectories,
            max_trajectory_length=int(max_trajectory_length),
            occupancy_trajectories_per_pool=target_occ_per_pool,
            occupancy_seed=int(seed) + 193_001,
            batch_size=target_evaluation_batch_size,
        )
        target_value = stopped_eval["value"]
        target_value_se = stopped_eval["value_se"]
        target_value_status = "coverage_stopped_mc_rollout_ok"
        target_occupancy_mass = stopped_eval["mass"]
        stopped_mass_se = stopped_eval["mass_se"]
        target_occ_states = stopped_eval["occupancy_states"]
        target_occ_actions = stopped_eval["occupancy_actions"]
        target_occ_episode_ids = stopped_eval["occupancy_episode_ids"]
        target_occ_pool_ids = stopped_eval["occupancy_pool_ids"]
    elif target_occ_per_pool > 0 and setting == "dice_rl_cartpole":
        target_evaluation_batch_size = max(int(collection_batch_size), 256)
        target_occ = _evaluate_infinite_cartpole_target_rollouts(
            target_policy=target_policy,
            gate_policy=None,
            action_encoder=action_encoder,
            gamma=float(gamma),
            seed=int(seed) + 191_001,
            num_trajectories=2 * target_occ_per_pool,
            max_trajectory_length=int(max_trajectory_length),
            occupancy_trajectories_per_pool=target_occ_per_pool,
            occupancy_seed=int(seed) + 193_001,
            batch_size=target_evaluation_batch_size,
        )
        target_occ_states = target_occ["occupancy_states"]
        target_occ_actions = target_occ["occupancy_actions"]
        target_occ_episode_ids = target_occ["occupancy_episode_ids"]
        target_occ_pool_ids = target_occ["occupancy_pool_ids"]
    elif target_occ_per_pool > 0:
        target_episodes = _collect_dice_rl_episodes(
            load_dir=load_dir,
            env_name=env_name,
            tabular_obs=tabular_obs,
            alpha=1.0,
            gamma=float(gamma),
            seed=int(seed) + 191_001,
            num_trajectories=2 * target_occ_per_pool,
            max_trajectory_length=int(max_trajectory_length),
            batch_size=int(collection_batch_size),
        )
        target_occ = sample_absorbing_discounted_occupancy(
            states=_flatten_features(target_episodes.states_raw),
            actions=action_encoder.encode(target_episodes.actions_raw),
            next_states=_flatten_features(target_episodes.next_states_raw),
            rewards=target_episodes.rewards,
            episode_ids=target_episodes.episode_ids,
            timesteps=target_episodes.timesteps,
            gamma=float(gamma),
            sample_size=2 * target_occ_per_pool,
            rng=np.random.default_rng(int(seed) + 193_001),
            replace_episodes=False,
        )
        target_occ_states = target_occ.states
        target_occ_actions = target_occ.actions
        target_occ_episode_ids = np.arange(2 * target_occ_per_pool, dtype=np.int64)
        target_occ_pool_ids = np.repeat(np.arange(2, dtype=np.int64), target_occ_per_pool)

    return BenchmarkDataset(
        setting=setting,
        states=occupancy.states,
        actions=occupancy.actions,
        next_states=occupancy.next_states,
        target_actions=target_actions,
        next_target_actions=next_target_actions,
        rewards=occupancy.rewards,
        true_ratio=None,
        true_action_ratio=None,
        true_transition_ratio=None,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=occupancy.masks,
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        episode_ids=np.asarray(occupancy.episode_ids, dtype=np.int64),
        timesteps=occupancy.timesteps,
        initial_episode_ids=behavior.initial_episode_ids.astype(np.int64, copy=False),
        is_absorbing=occupancy.is_absorbing,
        next_retention=next_retention,
        initial_retention=initial_retention,
        target_occupancy_states=target_occ_states,
        target_occupancy_actions=target_occ_actions,
        target_occupancy_episode_ids=target_occ_episode_ids,
        target_occupancy_pool_ids=target_occ_pool_ids,
        target_occupancy_mass=target_occupancy_mass,
        target_policy_value=target_value if np.isfinite(target_value) else None,
        target_policy_value_se=target_value_se if np.isfinite(target_value_se) else None,
        target_policy_value_kind=(
            "coverage_stopped_normalized_discounted_step_reward"
            if stopped_gate_track
            else "normalized_discounted_step_reward"
        ),
        metadata={
            "benchmark_track": (
                "known_greedy_gate_coverage_stopped_mechanism"
                if stopped_gate_track
                else "dualdice_paper_repro"
            ),
            "dataset_variant": f"alpha={alpha:g}",
            "dice_rl_env_name": env_name,
            "dice_rl_behavior_alpha": behavior_alpha,
            "dice_rl_target_alpha": target_alpha,
            "coverage_gate": "known_released_policy_greedy_action_gate" if stopped_gate_track else "full_support",
            "coverage_stopped_target": float(stopped_gate_track),
            "target_retained_occupancy_mass": (
                float(target_occupancy_mass) if target_occupancy_mass is not None else 1.0
            ),
            "target_retained_occupancy_mass_se": stopped_mass_se,
            "dice_rl_policy_load_dir": str(load_dir),
            "num_trajectories": int(num_trajectories),
            "max_trajectory_length": int(max_trajectory_length),
            "collection_batch_size": int(collection_batch_size),
            "target_evaluation_batch_size": target_evaluation_batch_size,
            "target_policy_value": target_value if np.isfinite(target_value) else np.nan,
            "target_policy_value_se": target_value_se if np.isfinite(target_value_se) else np.nan,
            "target_policy_value_kind": (
                "coverage_stopped_normalized_discounted_step_reward"
                if stopped_gate_track
                else "normalized_discounted_step_reward"
            ),
            "target_value_rollouts": actual_target_value_rollouts,
            "target_value_rollouts_requested": int(target_value_rollouts),
            "target_value_status": target_value_status,
            "truth_source": "target_policy_mc_rollout",
            "reference_distribution": "normalized_discounted_behavior_occupancy_with_absorbing_tail",
            "reference_sampling": "uniform_episode_geometric_time_with_absorbing_tail",
            "dice_rl_reacher_backend": "gymnasium_Reacher-v4_unwrapped_infinite_compat" if setting == "dice_rl_reacher" else "",
            "fold_grouping": "episode",
            "state_dim": int(occupancy.states.shape[1]),
            "action_dim": int(action_encoder.dim),
            "has_ratio_truth": 0.0,
            "absorbing_state_contract": 1.0,
            "absorbing_source_fraction": float(np.mean(occupancy.is_absorbing)),
            "target_occupancy_trajectories_per_pool": target_occ_per_pool,
            "target_occupancy_status": "mc_rollout_ok" if target_occ_per_pool > 0 else "disabled",
        },
    )


def _parse_alpha_variant(dataset_variant: str | None) -> float:
    if dataset_variant is None:
        return 0.0
    text = str(dataset_variant)
    if text.startswith("alpha="):
        text = text.split("=", maxsplit=1)[1]
    try:
        alpha = float(text)
    except ValueError as exc:
        raise ValueError(f"DICE-RL dataset variant must be 'alpha=<value>', got {dataset_variant!r}.") from exc
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("DICE-RL policy alpha must be in [0, 1].")
    return alpha


def _resolve_policy_load_dir(
    *,
    setting: str,
    dice_rl_repo_path: Path,
    asset_cache_dir: Path,
    install_assets: bool,
) -> Path:
    if setting in {
        "dice_rl_cartpole",
        "dice_rl_cartpole_stopped_greedy_gate",
        "dice_rl_reacher",
    }:
        testdata = dice_rl_repo_path / "tests" / "testdata"
        if testdata.exists():
            return testdata
        raise OptionalDatasetUnavailable(f"Missing DICE-RL test policy assets under {testdata}.")

    repo_policies = dice_rl_repo_path / "policies"
    if _has_taxi_assets(repo_policies):
        return repo_policies

    cache_policies = asset_cache_dir / "dice_rl" / "policies"
    if _has_taxi_assets(cache_policies):
        return cache_policies
    if install_assets:
        _install_taxi_assets(asset_cache_dir)
    if _has_taxi_assets(cache_policies):
        return cache_policies
    raise OptionalDatasetUnavailable(
        "Missing DICE-RL Taxi policy assets pi18.npy/pi19.npy after installation attempt."
    )


def _estimate_or_load_dice_rl_target_value(
    *,
    setting: str,
    load_dir: Path,
    env_name: str,
    tabular_obs: bool,
    gamma: float,
    seed: int,
    target_value_rollouts: int,
    max_trajectory_length: int,
    collection_batch_size: int,
    asset_cache_dir: Path,
) -> tuple[float, float, str]:
    cache_path = _target_value_cache_path(
        asset_cache_dir=asset_cache_dir,
        family="dice_rl",
        key_parts=(
            str(setting),
            str(env_name),
            f"tabular={int(bool(tabular_obs))}",
            f"gamma={float(gamma):.12g}",
            f"seed={int(seed)}",
            f"rollouts={int(target_value_rollouts)}",
            f"max={int(max_trajectory_length)}",
            f"batch={int(collection_batch_size)}",
        ),
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
    target_rollouts = _collect_dice_rl_episodes(
        load_dir=load_dir,
        env_name=env_name,
        tabular_obs=tabular_obs,
        alpha=1.0,
        gamma=float(gamma),
        seed=int(seed),
        num_trajectories=int(target_value_rollouts),
        max_trajectory_length=int(max_trajectory_length),
        batch_size=int(collection_batch_size),
    )
    target_values = np.asarray(target_rollouts.returns, dtype=np.float64)
    target_value = float(np.mean(target_values)) if target_values.size else float("nan")
    target_value_se = (
        float(np.std(target_values, ddof=1) / np.sqrt(target_values.size))
        if target_values.size > 1
        else float("nan")
    )
    status = "mc_rollout_ok" if np.isfinite(target_value) else "mc_rollout_nonfinite"
    if np.isfinite(target_value):
        _write_target_value_cache(
            cache_path,
            {
                "setting": str(setting),
                "env_name": str(env_name),
                "tabular_obs": bool(tabular_obs),
                "gamma": float(gamma),
                "seed": int(seed),
                "target_value_rollouts": int(target_value_rollouts),
                "max_trajectory_length": int(max_trajectory_length),
                "collection_batch_size": int(collection_batch_size),
                "target_value": float(target_value),
                "target_value_se": float(target_value_se) if np.isfinite(target_value_se) else None,
                "status": status,
            },
        )
    return target_value, target_value_se, status


def _target_value_cache_path(*, asset_cache_dir: Path, family: str, key_parts: tuple[str, ...]) -> Path:
    token = hashlib.sha1("|".join(str(part) for part in key_parts).encode("utf-8")).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(key_parts[0]))
    return asset_cache_dir / str(family) / "target_values" / f"{safe}_{token}.json"


def _write_target_value_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _has_taxi_assets(root: Path) -> bool:
    return (root / "taxi" / "pi18.npy").exists() and (root / "taxi" / "pi19.npy").exists()


def _install_taxi_assets(asset_cache_dir: Path) -> None:
    asset_cache_dir.mkdir(parents=True, exist_ok=True)
    source_root = asset_cache_dir / "infinite-horizon-off-policy-estimation"
    if not source_root.exists():
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/zt95/infinite-horizon-off-policy-estimation.git",
                    str(source_root),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        except Exception as exc:
            raise OptionalDatasetUnavailable(f"Taxi policy asset clone failed: {type(exc).__name__}: {exc}") from exc
    source = source_root / "taxi" / "taxi-policy"
    if not source.exists():
        raise OptionalDatasetUnavailable(f"Taxi policy asset clone did not contain {source}.")
    target = asset_cache_dir / "dice_rl" / "policies" / "taxi"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def _patch_legacy_gym_seed() -> None:
    """Restore legacy Gym APIs expected by DICE-RL/TF-Agents."""

    try:
        import gym  # noqa: PLC0415
    except Exception:
        return
    env_cls = getattr(gym, "Env", None)
    if env_cls is not None and not hasattr(env_cls, "seed"):

        def seed(self: Any, seed: int | None = None) -> list[int | None]:
            try:
                self.reset(seed=seed)
            except TypeError:
                self.reset()
            for space_name in ("action_space", "observation_space"):
                space = getattr(self, space_name, None)
                if hasattr(space, "seed"):
                    space.seed(seed)
            return [seed]

        setattr(env_cls, "seed", seed)
    _patch_cartpole_step_for_dice_rl()

    try:
        from tf_agents.environments import gym_wrapper  # noqa: PLC0415
        from tf_agents.trajectories import time_step as ts  # noqa: PLC0415
    except Exception:
        return
    wrapper_cls = getattr(gym_wrapper, "GymWrapper", None)
    if wrapper_cls is None or getattr(wrapper_cls, "_occupancy_ratio_gym_compat", False):
        return

    def _reset(self: Any) -> Any:
        result = self._gym_env.reset()
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            observation, info = result
            self._info = info
        else:
            observation = result
            self._info = None
        self._done = False
        if self._match_obs_space_dtype:
            observation = self._to_obs_space_dtype(observation)
        return ts.restart(observation)

    def _step(self: Any, action: Any) -> Any:
        if self._action_is_discrete and isinstance(action, np.ndarray):
            action = action.item()
        result = self._gym_env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, terminated, truncated, info = result
            self._done = bool(terminated or truncated)
            self._info = info
        else:
            observation, reward, self._done, self._info = result
        if self._match_obs_space_dtype:
            observation = self._to_obs_space_dtype(observation)
        if self._done:
            return ts.termination(observation, reward)
        return ts.transition(observation, reward, self._discount)

    setattr(wrapper_cls, "_reset", _reset)
    setattr(wrapper_cls, "_step", _step)
    setattr(wrapper_cls, "_occupancy_ratio_gym_compat", True)


def _patch_cartpole_step_for_dice_rl() -> None:
    try:
        from gym.envs.classic_control.cartpole import CartPoleEnv  # noqa: PLC0415
    except Exception:
        return
    if getattr(CartPoleEnv, "_occupancy_ratio_dice_rl_step_compat", False):
        return
    original_step = CartPoleEnv.step

    def step(self: Any, action: Any) -> Any:
        result = original_step(self, action)
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return observation, reward, bool(terminated or truncated), info
        return result

    setattr(CartPoleEnv, "step", step)
    setattr(CartPoleEnv, "_occupancy_ratio_dice_rl_step_compat", True)


def _collect_dice_rl_episodes(
    *,
    load_dir: Path,
    env_name: str,
    tabular_obs: bool,
    alpha: float,
    gamma: float,
    seed: int,
    num_trajectories: int,
    max_trajectory_length: int,
    batch_size: int = 20,
) -> _DiceRLEpisodes:
    import tensorflow as tf  # noqa: PLC0415
    from dice_rl.data import tf_agents_onpolicy_dataset  # noqa: PLC0415
    from dice_rl.data.dataset import StepType  # noqa: PLC0415

    env_policies = _import_dice_rl_env_policies(env_name)
    _patch_dice_rl_gridworld_seeding()
    _patch_dice_rl_reacher_backend(env_policies)
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    tf_env, policy = env_policies.get_env_and_policy(
        str(load_dir),
        str(env_name),
        float(alpha),
        env_seed=int(seed),
        tabular_obs=bool(tabular_obs),
    )
    if str(env_name) == "cartpole" and not bool(tabular_obs):
        return _collect_infinite_cartpole_episodes_batched(
            policy=policy,
            action_spec=tf_env.action_spec(),
            gamma=float(gamma),
            seed=int(seed),
            num_trajectories=int(num_trajectories),
            max_trajectory_length=int(max_trajectory_length),
            batch_size=int(batch_size),
        )
    dataset = tf_agents_onpolicy_dataset.TFAgentsOnpolicyDataset(
        tf_env,
        policy,
        episode_step_limit=int(max_trajectory_length),
    )
    action_spec = tf_env.action_spec()
    states: list[Array] = []
    actions: list[Array] = []
    next_states: list[Array] = []
    rewards: list[float] = []
    masks: list[float] = []
    timesteps: list[int] = []
    episode_ids: list[int] = []
    initial_states: list[Array] = []
    initial_episode_ids: list[int] = []
    returns: list[float] = []

    collected = 0
    batch_size = min(max(1, int(batch_size)), max(1, int(num_trajectories)))
    while collected < int(num_trajectories):
        current = min(batch_size, int(num_trajectories) - collected)
        episodes, valid_steps = dataset.get_episode(batch_size=current)
        valid_np = np.asarray(_to_numpy(valid_steps), dtype=bool)
        obs_np = np.asarray(_to_numpy(episodes.observation))
        action_np = np.asarray(_to_numpy(episodes.action))
        reward_np = np.asarray(_to_numpy(episodes.reward), dtype=np.float64)
        step_type_np = np.asarray(_to_numpy(episodes.step_type))
        if valid_np.ndim == 1:
            valid_np = valid_np.reshape(1, -1)
            obs_np = obs_np.reshape((1,) + obs_np.shape)
            action_np = action_np.reshape((1,) + action_np.shape)
            reward_np = reward_np.reshape((1,) + reward_np.shape)
            step_type_np = step_type_np.reshape((1,) + step_type_np.shape)
        for local_ep in range(valid_np.shape[0]):
            ep_id = collected + local_ep
            valid_idx = np.flatnonzero(valid_np[local_ep])
            if valid_idx.size < 2:
                continue
            obs = obs_np[local_ep, valid_idx]
            act = action_np[local_ep, valid_idx]
            rew = reward_np[local_ep, valid_idx].reshape(-1)
            step_type = step_type_np[local_ep, valid_idx].reshape(-1)
            initial_states.append(np.asarray(obs[0]).copy())
            initial_episode_ids.append(int(ep_id))
            ep_return = 0.0
            for t in range(valid_idx.size - 1):
                states.append(np.asarray(obs[t]).copy())
                actions.append(np.asarray(act[t]).copy())
                next_states.append(np.asarray(obs[t + 1]).copy())
                reward = float(rew[t])
                rewards.append(reward)
                masks.append(float(not StepType.is_last(step_type[t + 1])))
                timesteps.append(int(t))
                episode_ids.append(int(ep_id))
                ep_return += (float(gamma) ** int(t)) * reward
            returns.append(float((1.0 - float(gamma)) * ep_return))
        collected += current

    states_arr = _stack_or_empty(states)
    return _DiceRLEpisodes(
        states_raw=states_arr,
        actions_raw=_stack_or_empty(actions),
        next_states_raw=_stack_or_empty(next_states),
        rewards=np.asarray(rewards, dtype=np.float64),
        masks=np.asarray(masks, dtype=np.float64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        initial_states_raw=_stack_or_empty(initial_states),
        initial_episode_ids=np.asarray(initial_episode_ids, dtype=np.int64),
        returns=np.asarray(returns, dtype=np.float64),
        action_spec=action_spec,
    )


def _collect_infinite_cartpole_episodes_batched(
    *,
    policy: Any,
    action_spec: Any,
    gamma: float,
    seed: int,
    num_trajectories: int,
    max_trajectory_length: int,
    batch_size: int,
) -> _DiceRLEpisodes:
    """Collect DICE-RL CartPole trajectories with batched policy evaluation."""

    from dice_rl.environments.infinite_cartpole import InfiniteCartPole  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    rng = np.random.RandomState(int(seed))
    states: list[Array] = []
    actions: list[Array] = []
    next_states: list[Array] = []
    rewards: list[float] = []
    masks: list[float] = []
    timesteps: list[int] = []
    episode_ids: list[int] = []
    initial_states: list[Array] = []
    initial_episode_ids: list[int] = []
    returns: list[float] = []

    collected = 0
    batch_size = min(max(1, int(batch_size)), max(1, int(num_trajectories)))
    while collected < int(num_trajectories):
        current = min(batch_size, int(num_trajectories) - collected)
        envs = [InfiniteCartPole() for _ in range(current)]
        obs = rng.uniform(low=-0.05, high=0.05, size=(current, 4)).astype(np.float32)
        episode_returns = np.zeros(current, dtype=np.float64)
        for local_ep, env in enumerate(envs):
            env.state = np.asarray(obs[local_ep], dtype=np.float32).copy()
            for boundary_attr in ("steps_beyond_done", "steps_beyond_terminated"):
                if hasattr(env, boundary_attr):
                    setattr(env, boundary_attr, None)
            initial_states.append(np.asarray(obs[local_ep], dtype=np.float64).copy())
            initial_episode_ids.append(int(collected + local_ep))

        for t in range(int(max_trajectory_length)):
            action_batch = np.asarray(_policy_actions(policy, obs)).reshape(-1)
            next_obs = np.empty_like(obs)
            for local_ep, env in enumerate(envs):
                ep_id = int(collected + local_ep)
                action = int(action_batch[local_ep])
                result = env.step(action)
                if isinstance(result, tuple) and len(result) == 5:
                    obs_next, reward, terminated, truncated, _info = result
                    done = bool(terminated or truncated)
                else:
                    obs_next, reward, done, _info = result
                del done
                for boundary_attr in ("steps_beyond_done", "steps_beyond_terminated"):
                    if hasattr(env, boundary_attr):
                        setattr(env, boundary_attr, None)
                obs_next = np.asarray(obs_next, dtype=np.float32).reshape(-1)
                states.append(np.asarray(obs[local_ep], dtype=np.float64).copy())
                actions.append(np.asarray(action).copy())
                next_states.append(np.asarray(obs_next, dtype=np.float64).copy())
                reward_value = float(reward)
                rewards.append(reward_value)
                masks.append(float(t + 1 < int(max_trajectory_length)))
                timesteps.append(int(t))
                episode_ids.append(ep_id)
                episode_returns[local_ep] += (float(gamma) ** int(t)) * reward_value
                next_obs[local_ep] = obs_next
            obs = next_obs
        returns.extend(float((1.0 - float(gamma)) * value) for value in episode_returns)
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        collected += current

    states_arr = _stack_or_empty(states)
    return _DiceRLEpisodes(
        states_raw=states_arr,
        actions_raw=_stack_or_empty(actions),
        next_states_raw=_stack_or_empty(next_states),
        rewards=np.asarray(rewards, dtype=np.float64),
        masks=np.asarray(masks, dtype=np.float64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        initial_states_raw=_stack_or_empty(initial_states),
        initial_episode_ids=np.asarray(initial_episode_ids, dtype=np.int64),
        returns=np.asarray(returns, dtype=np.float64),
        action_spec=action_spec,
    )


def _load_dice_rl_policy(
    *,
    load_dir: Path,
    env_name: str,
    tabular_obs: bool,
    alpha: float,
    seed: int,
):
    env_policies = _import_dice_rl_env_policies(env_name)
    _patch_dice_rl_gridworld_seeding()
    _patch_dice_rl_reacher_backend(env_policies)
    _, policy = env_policies.get_env_and_policy(
        str(load_dir),
        str(env_name),
        float(alpha),
        env_seed=int(seed),
        tabular_obs=bool(tabular_obs),
    )
    return policy


def _import_dice_rl_env_policies(env_name: str) -> Any:
    """Import DICE-RL policies without loading unused legacy MuJoCo on classic control."""

    if str(env_name) != "reacher":
        reacher_module = "dice_rl.environments.infinite_reacher"
        if reacher_module not in sys.modules:
            stub = types.ModuleType(reacher_module)

            class UnusedInfiniteReacher:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    del args, kwargs
                    raise RuntimeError("InfiniteReacher is unavailable in this non-Reacher benchmark process.")

            stub.InfiniteReacher = UnusedInfiniteReacher
            sys.modules[reacher_module] = stub
        mujoco_suite_module = "tf_agents.environments.suite_mujoco"
        if mujoco_suite_module not in sys.modules:
            suite_stub = types.ModuleType(mujoco_suite_module)

            def unavailable_mujoco(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                raise RuntimeError("TF-Agents MuJoCo is unavailable in this non-MuJoCo benchmark process.")

            suite_stub.load = unavailable_mujoco
            sys.modules[mujoco_suite_module] = suite_stub
    return importlib.import_module("dice_rl.environments.env_policies")


def _patch_dice_rl_gridworld_seeding() -> None:
    """Use old RandomState-style RNGs for DICE-RL custom tabular environments."""

    def np_random(seed: int | None = None) -> tuple[np.random.RandomState, int]:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        return np.random.RandomState(int(seed)), int(seed)

    module_names = (
        "dice_rl.environments.gridworld.taxi",
        "dice_rl.environments.gridworld.navigation",
        "dice_rl.environments.gridworld.point_maze",
        "dice_rl.environments.gridworld.maze",
        "dice_rl.environments.gridworld.tree",
        "dice_rl.environments.gridworld.low_rank",
        "dice_rl.environments.bandit",
        "dice_rl.environments.bernoulli_bandit",
        "dice_rl.environments.contextual_bandit",
        "dice_rl.environments.line",
    )
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["seeding"])
            seeding = getattr(module, "seeding", None)
            if seeding is not None:
                setattr(seeding, "np_random", np_random)
        except Exception:
            continue


def _patch_dice_rl_reacher_backend(env_policies: Any) -> None:
    backend_name = "gymnasium_Reacher-v4_unwrapped_infinite_compat"
    if getattr(env_policies, "_occupancy_ratio_reacher_backend", "") == backend_name:
        return
    try:
        import gym  # noqa: PLC0415
        import gymnasium as gymnasium  # noqa: PLC0415
    except Exception:
        return

    class GymnasiumInfiniteReacher(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            try:
                from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv as GymnasiumReacherEnv  # noqa: PLC0415

                self._env = GymnasiumReacherEnv()
            except Exception:
                self._env = gymnasium.make("Reacher-v4").unwrapped
            self.action_space = gym.spaces.Box(
                low=np.asarray(self._env.action_space.low, dtype=np.float32),
                high=np.asarray(self._env.action_space.high, dtype=np.float32),
                shape=tuple(self._env.action_space.shape),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Box(
                low=np.asarray(self._env.observation_space.low, dtype=np.float32),
                high=np.asarray(self._env.observation_space.high, dtype=np.float32),
                shape=tuple(self._env.observation_space.shape),
                dtype=np.float32,
            )
            self.np_random = np.random.RandomState(0)

        def seed(self, seed: int | None = None) -> list[int | None]:
            self.np_random = np.random.RandomState(0 if seed is None else int(seed))
            try:
                self.action_space.seed(seed)
                self.observation_space.seed(seed)
            except Exception:
                pass
            return [seed]

        def reset(self, **kwargs: Any) -> Array:
            seed = kwargs.get("seed")
            if seed is None:
                seed = int(self.np_random.randint(2**31 - 1))
            out = self._env.reset(seed=int(seed))
            obs = out[0] if isinstance(out, tuple) else out
            return np.asarray(obs, dtype=np.float32)

        def step(self, action: Array) -> tuple[Array, float, bool, dict[str, Any]]:
            act = np.clip(
                np.asarray(action, dtype=np.float32),
                np.asarray(self.action_space.low, dtype=np.float32),
                np.asarray(self.action_space.high, dtype=np.float32),
            )
            out = self._env.step(act)
            if len(out) == 5:
                obs, reward, _terminated, _truncated, info = out
            else:
                obs, reward, _done, info = out
            if self.np_random.rand() < 0.03:
                obs = self.reset()
            return np.asarray(obs, dtype=np.float32), float(reward), False, dict(info)

        def close(self) -> None:
            self._env.close()

    setattr(env_policies, "InfiniteReacher", GymnasiumInfiniteReacher)
    setattr(env_policies, "_occupancy_ratio_reacher_backend", backend_name)


def _policy_actions(policy: Any, observations: Array) -> Array:
    import tensorflow as tf  # noqa: PLC0415
    from tf_agents.trajectories import time_step  # noqa: PLC0415

    obs = np.asarray(observations)
    try:
        spec_dtype = policy.time_step_spec.observation.dtype
        obs = obs.astype(np.dtype(spec_dtype.as_numpy_dtype if hasattr(spec_dtype, "as_numpy_dtype") else spec_dtype), copy=False)
    except Exception:
        pass
    n = int(obs.shape[0])
    observation = tf.convert_to_tensor(obs)
    ts = time_step.TimeStep(
        step_type=tf.fill((n,), time_step.StepType.MID),
        reward=tf.zeros((n,), dtype=tf.float32),
        discount=tf.ones((n,), dtype=tf.float32),
        observation=observation,
    )
    return np.asarray(policy.action(ts).action.numpy())


def _policy_actions_batched(policy: Any, observations: Array, *, batch_size: int = 8192) -> Array:
    obs = np.asarray(observations)
    if obs.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    chunks = [
        _policy_actions(policy, obs[start : start + int(batch_size)])
        for start in range(0, obs.shape[0], int(batch_size))
    ]
    return np.concatenate(chunks, axis=0)


def _policy_mode_actions(policy: Any, observations: Array) -> Array:
    """Return deterministic modal actions, including through epsilon-greedy wrappers."""

    import tensorflow as tf  # noqa: PLC0415
    from tf_agents.trajectories import time_step  # noqa: PLC0415

    obs = np.asarray(observations)
    try:
        spec_dtype = policy.time_step_spec.observation.dtype
        obs = obs.astype(
            np.dtype(spec_dtype.as_numpy_dtype if hasattr(spec_dtype, "as_numpy_dtype") else spec_dtype),
            copy=False,
        )
    except Exception:
        pass
    n = int(obs.shape[0])
    ts = time_step.TimeStep(
        step_type=tf.fill((n,), time_step.StepType.MID),
        reward=tf.zeros((n,), dtype=tf.float32),
        discount=tf.ones((n,), dtype=tf.float32),
        observation=tf.convert_to_tensor(obs),
    )
    distribution = policy.distribution(ts).action
    return np.asarray(distribution.mode().numpy())


def _policy_mode_actions_batched(policy: Any, observations: Array, *, batch_size: int = 8192) -> Array:
    obs = np.asarray(observations)
    if obs.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    chunks = [
        _policy_mode_actions(policy, obs[start : start + int(batch_size)])
        for start in range(0, obs.shape[0], int(batch_size))
    ]
    return np.concatenate(chunks, axis=0)


def _encoded_action_match(left: Array, right: Array) -> Array:
    lhs = np.asarray(left, dtype=np.float64).reshape(np.asarray(left).shape[0], -1)
    rhs = np.asarray(right, dtype=np.float64).reshape(np.asarray(right).shape[0], -1)
    if lhs.shape != rhs.shape:
        raise ValueError("Encoded actions must have matching shapes for support checks.")
    return np.all(np.isclose(lhs, rhs, rtol=0.0, atol=1e-8), axis=1).astype(np.float64)


def _evaluate_infinite_cartpole_target_rollouts(
    *,
    target_policy: Any,
    gate_policy: Any | None,
    action_encoder: "_ActionEncoder",
    gamma: float,
    seed: int,
    num_trajectories: int,
    max_trajectory_length: int,
    occupancy_trajectories_per_pool: int,
    occupancy_seed: int,
    batch_size: int,
) -> dict[str, Any]:
    """Stream CartPole value and geometric-time occupancy without storing trajectories."""

    from dice_rl.environments.infinite_cartpole import InfiniteCartPole  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    n_trajectories = int(num_trajectories)
    horizon = int(max_trajectory_length)
    per_pool = int(occupancy_trajectories_per_pool)
    if n_trajectories <= 0 or horizon <= 0 or per_pool < 0:
        raise ValueError("CartPole evaluation sizes must be positive and occupancy pools nonnegative.")
    occupancy_count = min(2 * per_pool, n_trajectories)
    gamma_f = float(gamma)
    if not 0.0 <= gamma_f < 1.0:
        raise ValueError("gamma must lie in [0, 1).")

    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    state_rng = np.random.RandomState(int(seed))
    occupancy_rng = np.random.default_rng(int(occupancy_seed))
    sampled_times = (
        occupancy_rng.geometric(1.0 - gamma_f, size=occupancy_count).astype(np.int64) - 1
        if gamma_f > 0.0
        else np.zeros(occupancy_count, dtype=np.int64)
    )
    values = np.zeros(n_trajectories, dtype=np.float64)
    masses = np.ones(n_trajectories, dtype=np.float64)
    occupancy_states: list[Array] = []
    occupancy_actions: list[Array] = []
    occupancy_episode_ids: list[int] = []
    occupancy_pool_ids: list[int] = []

    effective_batch_size = min(max(1, int(batch_size)), n_trajectories)
    for start in range(0, n_trajectories, effective_batch_size):
        current = min(effective_batch_size, n_trajectories - start)
        envs = [InfiniteCartPole() for _ in range(current)]
        observations = state_rng.uniform(low=-0.05, high=0.05, size=(current, 4)).astype(np.float32)
        alive = np.ones(current, dtype=bool)
        discounted_returns = np.zeros(current, dtype=np.float64)
        for local_ep, env in enumerate(envs):
            env.state = observations[local_ep].copy()
            for boundary_attr in ("steps_beyond_done", "steps_beyond_terminated"):
                if hasattr(env, boundary_attr):
                    setattr(env, boundary_attr, None)

        for timestep in range(horizon):
            active = np.flatnonzero(alive)
            if active.size == 0:
                break
            target_raw = np.asarray(_policy_actions(target_policy, observations[active])).reshape(-1)
            target_encoded = action_encoder.encode(target_raw)
            if gate_policy is None:
                accepted = np.ones(active.size, dtype=bool)
            else:
                gate_raw = np.asarray(_policy_mode_actions(gate_policy, observations[active])).reshape(-1)
                gate_encoded = action_encoder.encode(gate_raw)
                accepted = _encoded_action_match(target_encoded, gate_encoded).astype(bool)

            rejected_local = active[~accepted]
            if rejected_local.size:
                masses[start + rejected_local] = 1.0 - gamma_f**int(timestep)
                alive[rejected_local] = False

            accepted_positions = np.flatnonzero(accepted)
            included_local = active[accepted_positions]
            for position, local_ep in zip(accepted_positions, included_local, strict=True):
                global_ep = start + int(local_ep)
                if global_ep < occupancy_count and int(sampled_times[global_ep]) == int(timestep):
                    occupancy_states.append(
                        np.concatenate(
                            [np.asarray(observations[local_ep], dtype=np.float64), np.array([0.0])]
                        )
                    )
                    occupancy_actions.append(np.asarray(target_encoded[position], dtype=np.float64))
                    occupancy_episode_ids.append(global_ep)
                    occupancy_pool_ids.append(min(global_ep // max(per_pool, 1), 1))

            for position, local_ep in zip(accepted_positions, included_local, strict=True):
                env = envs[int(local_ep)]
                result = env.step(int(target_raw[position]))
                if isinstance(result, tuple) and len(result) == 5:
                    next_observation, reward, _terminated, _truncated, _info = result
                else:
                    next_observation, reward, _done, _info = result
                for boundary_attr in ("steps_beyond_done", "steps_beyond_terminated"):
                    if hasattr(env, boundary_attr):
                        setattr(env, boundary_attr, None)
                observations[local_ep] = np.asarray(next_observation, dtype=np.float32).reshape(-1)
                discounted_returns[local_ep] += gamma_f**int(timestep) * float(reward)

        values[start : start + current] = (1.0 - gamma_f) * discounted_returns
        for local_ep in np.flatnonzero(alive):
            global_ep = start + int(local_ep)
            if global_ep < occupancy_count and int(sampled_times[global_ep]) >= horizon:
                absorbing_state = np.zeros(observations.shape[1] + 1, dtype=np.float64)
                absorbing_state[-1] = 1.0
                occupancy_states.append(absorbing_state)
                occupancy_actions.append(np.zeros(action_encoder.dim, dtype=np.float64))
                occupancy_episode_ids.append(global_ep)
                occupancy_pool_ids.append(min(global_ep // max(per_pool, 1), 1))
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    state_dim = 5
    return {
        "value": float(np.mean(values)),
        "value_se": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
        "mass": float(np.mean(masses)),
        "mass_se": float(np.std(masses, ddof=1) / np.sqrt(masses.size)) if masses.size > 1 else 0.0,
        "num_trajectories": n_trajectories,
        "occupancy_states": (
            np.asarray(occupancy_states, dtype=np.float64)
            if occupancy_states
            else np.empty((0, state_dim), dtype=np.float64)
        ),
        "occupancy_actions": (
            np.asarray(occupancy_actions, dtype=np.float64)
            if occupancy_actions
            else np.empty((0, action_encoder.dim), dtype=np.float64)
        ),
        "occupancy_episode_ids": np.asarray(occupancy_episode_ids, dtype=np.int64),
        "occupancy_pool_ids": np.asarray(occupancy_pool_ids, dtype=np.int64),
    }


def _evaluate_known_gate_stopping(
    *,
    episodes: _DiceRLEpisodes,
    gate_policy: Any,
    action_encoder: "_ActionEncoder",
    gamma: float,
    occupancy_trajectories_per_pool: int,
    seed: int,
) -> dict[str, Any]:
    gate_actions = action_encoder.encode(
        _policy_mode_actions_batched(gate_policy, episodes.states_raw)
    )
    observed_actions = action_encoder.encode(episodes.actions_raw)
    accepted = _encoded_action_match(observed_actions, gate_actions).astype(bool)
    episode_ids = np.asarray(episodes.episode_ids, dtype=np.int64).reshape(-1)
    timesteps = np.asarray(episodes.timesteps, dtype=np.int64).reshape(-1)
    rewards = np.asarray(episodes.rewards, dtype=np.float64).reshape(-1)
    initial_ids = np.asarray(episodes.initial_episode_ids, dtype=np.int64).reshape(-1)
    value_rows: list[float] = []
    mass_rows: list[float] = []
    first_rejection: dict[int, int | None] = {}
    row_lookup: dict[tuple[int, int], int] = {}
    episode_length: dict[int, int] = {}
    for row, (episode_id, timestep) in enumerate(zip(episode_ids, timesteps, strict=True)):
        row_lookup[(int(episode_id), int(timestep))] = int(row)
        episode_length[int(episode_id)] = max(episode_length.get(int(episode_id), 0), int(timestep) + 1)

    gamma_f = float(gamma)
    for episode_id in initial_ids:
        ep = int(episode_id)
        length = int(episode_length.get(ep, 0))
        total = 0.0
        rejected_at: int | None = None
        for timestep in range(length):
            row = row_lookup[(ep, timestep)]
            if not bool(accepted[row]):
                rejected_at = int(timestep)
                break
            total += (gamma_f**timestep) * float(rewards[row])
        value_rows.append((1.0 - gamma_f) * total)
        mass_rows.append(1.0 if rejected_at is None else 1.0 - gamma_f**rejected_at)
        first_rejection[ep] = rejected_at

    values = np.asarray(value_rows, dtype=np.float64)
    masses = np.asarray(mass_rows, dtype=np.float64)
    result: dict[str, Any] = {
        "value": float(np.mean(values)),
        "value_se": (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        ),
        "mass": float(np.mean(masses)),
        "mass_se": (
            float(np.std(masses, ddof=1) / np.sqrt(masses.size))
            if masses.size > 1
            else 0.0
        ),
        "num_trajectories": int(values.size),
        "occupancy_states": None,
        "occupancy_actions": None,
        "occupancy_episode_ids": None,
        "occupancy_pool_ids": None,
    }
    per_pool = int(occupancy_trajectories_per_pool)
    if per_pool <= 0:
        return result

    rng = np.random.default_rng(int(seed))
    selected_ids = initial_ids[: 2 * per_pool]
    occupancy_states: list[Array] = []
    occupancy_actions: list[Array] = []
    occupancy_episode_ids: list[int] = []
    occupancy_pool_ids: list[int] = []
    state_dim = int(_flatten_features(episodes.states_raw).shape[1])
    for position, episode_id in enumerate(selected_ids):
        ep = int(episode_id)
        sampled_time = int(rng.geometric(1.0 - gamma_f) - 1) if gamma_f > 0.0 else 0
        rejected_at = first_rejection[ep]
        if rejected_at is not None and sampled_time >= rejected_at:
            continue
        length = int(episode_length.get(ep, 0))
        if sampled_time < length:
            row = row_lookup[(ep, sampled_time)]
            state = np.concatenate(
                [_flatten_features(episodes.states_raw[row : row + 1])[0], np.array([0.0])]
            )
            action = observed_actions[row]
        else:
            state = np.zeros(state_dim + 1, dtype=np.float64)
            state[-1] = 1.0
            action = np.zeros(action_encoder.dim, dtype=np.float64)
        occupancy_states.append(state)
        occupancy_actions.append(np.asarray(action, dtype=np.float64))
        occupancy_episode_ids.append(ep)
        occupancy_pool_ids.append(min(position // per_pool, 1))
    result.update(
        {
            "occupancy_states": np.asarray(occupancy_states, dtype=np.float64),
            "occupancy_actions": np.asarray(occupancy_actions, dtype=np.float64),
            "occupancy_episode_ids": np.asarray(occupancy_episode_ids, dtype=np.int64),
            "occupancy_pool_ids": np.asarray(occupancy_pool_ids, dtype=np.int64),
        }
    )
    return result


def _to_numpy(value: Any) -> Array:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _stack_or_empty(values: list[Array]) -> Array:
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(values, axis=0)


def _flatten_features(values: Array) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)


@dataclass(frozen=True)
class _ActionEncoder:
    dim: int
    discrete: bool
    offset: int = 0

    @classmethod
    def from_spec(cls, spec: Any) -> "_ActionEncoder":
        dtype = getattr(spec, "dtype", None)
        np_dtype = np.dtype(dtype.as_numpy_dtype if hasattr(dtype, "as_numpy_dtype") else dtype)
        shape = tuple(getattr(spec, "shape", ()) or ())
        if np.issubdtype(np_dtype, np.integer) and shape == ():
            minimum = int(np.asarray(getattr(spec, "minimum", 0)).reshape(-1)[0])
            maximum = int(np.asarray(getattr(spec, "maximum", 1)).reshape(-1)[0])
            return cls(dim=maximum - minimum + 1, discrete=True, offset=minimum)
        dim = int(np.prod(shape or (1,)))
        return cls(dim=dim, discrete=False, offset=0)

    def encode(self, actions: Array) -> Array:
        arr = np.asarray(actions)
        if self.discrete:
            return one_hot(arr.astype(np.int64).reshape(-1) - int(self.offset), int(self.dim))
        return np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
