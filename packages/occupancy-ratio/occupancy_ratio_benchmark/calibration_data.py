"""Dataset construction and lossless caching for calibration units."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.discrete import make_discrete_dataset
from occupancy_ratio_benchmark.gaussian import make_linear_gaussian_dataset


Array = np.ndarray
SUPPORTED_FAMILIES = (
    "random_tabular",
    "linear_gaussian",
    "d4rl_matched",
    "dice_rl_cartpole",
)
_ARRAY_FIELDS = tuple(
    field.name
    for field in fields(BenchmarkDataset)
    if field.name
    not in {
        "setting",
        "gamma",
        "seed",
        "sample_size",
        "target_occupancy_mass",
        "target_policy_value",
        "target_policy_value_se",
        "target_policy_value_kind",
        "metadata",
    }
)


@dataclass(frozen=True)
class CalibrationDatasetBundle:
    """One normalized dataset plus fold/audit groups and truth telemetry."""

    dataset: BenchmarkDataset
    source_groups: Array
    initial_groups: Array
    truth_stages: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class DatasetPaths:
    """External assets required by realistic benchmark generators."""

    asset_cache: Path
    dice_rl: Path = Path("/Users/larsvanderlaan/repos/dice_rl")
    install_assets: bool = False


def build_calibration_dataset(
    *,
    cell: Mapping[str, Any],
    axis_values: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    paths: DatasetPaths,
) -> CalibrationDatasetBundle:
    """Build a strict manifest cell without importing stopped estimands."""

    family = str(_required(cell, "benchmark_family"))
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported calibration benchmark_family {family!r}")
    sample_size = _positive_int(_required(axis_values, "sample_size"), "sample_size")
    seed = _integer(_required(axis_values, "seed"), "seed")
    gamma = float(_required(axis_values, "gamma"))
    if not np.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be finite and in [0, 1)")

    truth_stages: tuple[dict[str, Any], ...] = ()
    if family == "random_tabular":
        dataset = make_discrete_dataset(
            setting="random_tabular_mdp",
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            policy_shift=float(_required(cell, "policy_shift")),
            n_states=_positive_int(_required(cell, "states"), "cell.states"),
            n_actions=_positive_int(_required(cell, "actions"), "cell.actions"),
        )
    elif family == "linear_gaussian":
        dataset = make_linear_gaussian_dataset(
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            policy_shift=float(_required(cell, "policy_shift")),
        )
    elif family == "d4rl_matched":
        from occupancy_ratio_benchmark.d4rl_ope import make_d4rl_ope_dataset  # noqa: PLC0415

        truth = _mapping(resolved_config, "truth")
        target_occupancy_trajectories_per_pool = _nonnegative_int(
            resolved_config.get("target_occupancy_trajectories_per_pool", 0),
            "target_occupancy_trajectories_per_pool",
        )

        def build(rollouts: int) -> BenchmarkDataset:
            audit = _independent_audit_config(resolved_config)
            return make_d4rl_ope_dataset(
                policy_id=str(_required(cell, "policy_id")),
                gamma=gamma,
                sample_size=sample_size,
                seed=seed,
                asset_cache_dir=paths.asset_cache,
                install_assets=bool(paths.install_assets),
                target_value_rollouts=int(rollouts),
                target_occupancy_trajectories_per_pool=target_occupancy_trajectories_per_pool,
                require_exact_rollout_env=True,
                behavior_episode_partition="train" if audit["enabled"] else "all",
                behavior_audit_fraction=float(audit["d4rl_raw_episode_fraction"]),
                behavior_partition_seed=int(audit["partition_seed"]),
            )

        dataset, truth_stages = _adaptive_truth_dataset(build, truth)
        _validate_target_occupancy_pools(
            dataset,
            trajectories_per_pool=target_occupancy_trajectories_per_pool,
        )
    else:
        from occupancy_ratio_benchmark.dice_rl_repro import (  # noqa: PLC0415
            make_dice_rl_reproduction_dataset,
        )

        collection = _mapping(resolved_config, "collection")
        truth = _mapping(resolved_config, "truth")
        if collection.get("coverage_gate") is not None or collection.get("full_support") is not True:
            raise ValueError("CartPole calibration requires the full-support no-gate collection")
        behavior_alpha = float(_required(cell, "behavior_alpha"))
        target_alpha = float(_required(cell, "target_alpha"))
        if target_alpha != 1.0:
            raise ValueError("the frozen CartPole target alpha is 1.0")

        def build(rollouts: int) -> BenchmarkDataset:
            return make_dice_rl_reproduction_dataset(
                setting="dice_rl_cartpole",
                dataset_variant=f"alpha={behavior_alpha:g}",
                gamma=gamma,
                sample_size=sample_size,
                seed=seed,
                dice_rl_repo_path=paths.dice_rl,
                asset_cache_dir=paths.asset_cache,
                install_assets=bool(paths.install_assets),
                num_trajectories=_positive_int(
                    _required(collection, "trajectories"), "collection.trajectories"
                ),
                max_trajectory_length=_positive_int(
                    _required(collection, "horizon"), "collection.horizon"
                ),
                target_value_rollouts=int(rollouts),
                target_occupancy_trajectories_per_pool=0,
                collection_batch_size=20,
            )

        dataset, truth_stages = _adaptive_truth_dataset(build, truth)

    validate_normalized_dataset(dataset)
    source_groups, initial_groups = calibration_group_ids(dataset)
    dataset.metadata = {
        **dataset.metadata,
        "calibration_benchmark_family": family,
        "calibration_truth_stages": list(truth_stages),
        "calibration_source_group_count": int(np.unique(source_groups).size),
        "calibration_initial_group_count": int(np.unique(initial_groups).size),
    }
    return CalibrationDatasetBundle(
        dataset=dataset,
        source_groups=source_groups,
        initial_groups=initial_groups,
        truth_stages=truth_stages,
    )


def build_calibration_audit_dataset(
    *,
    cell: Mapping[str, Any],
    axis_values: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    paths: DatasetPaths,
) -> CalibrationDatasetBundle:
    """Build behavior data excluded from every estimator and calibrator fit."""

    audit = _independent_audit_config(resolved_config)
    if not audit["enabled"]:
        raise ValueError("independent behavior audit is not enabled")
    family = str(_required(cell, "benchmark_family"))
    sample_size = _positive_int(_required(axis_values, "sample_size"), "sample_size")
    seed = _integer(_required(axis_values, "seed"), "seed")
    gamma = float(_required(axis_values, "gamma"))
    audit_seed = _audit_seed(
        config_id=str(_required(resolved_config, "config_id")),
        cell_id=str(_required(cell, "cell_id")),
        sample_size=sample_size,
        gamma=gamma,
        seed=seed,
        salt=int(audit["sample_seed_salt"]),
    )
    if family == "random_tabular":
        dataset = make_discrete_dataset(
            setting="random_tabular_mdp",
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            sample_seed=audit_seed,
            policy_shift=float(_required(cell, "policy_shift")),
            n_states=_positive_int(_required(cell, "states"), "cell.states"),
            n_actions=_positive_int(_required(cell, "actions"), "cell.actions"),
        )
    elif family == "linear_gaussian":
        dataset = make_linear_gaussian_dataset(
            gamma=gamma,
            sample_size=sample_size,
            seed=seed,
            sample_seed=audit_seed,
            policy_shift=float(_required(cell, "policy_shift")),
        )
    elif family == "d4rl_matched":
        from occupancy_ratio_benchmark.d4rl_ope import make_d4rl_ope_dataset  # noqa: PLC0415

        dataset = make_d4rl_ope_dataset(
            policy_id=str(_required(cell, "policy_id")),
            gamma=gamma,
            sample_size=sample_size,
            seed=audit_seed,
            asset_cache_dir=paths.asset_cache,
            install_assets=bool(paths.install_assets),
            target_value_rollouts=1,
            target_occupancy_trajectories_per_pool=0,
            require_exact_rollout_env=False,
            behavior_episode_partition="audit",
            behavior_audit_fraction=float(audit["d4rl_raw_episode_fraction"]),
            behavior_partition_seed=int(audit["partition_seed"]),
            include_target_evaluation=False,
        )
    elif family == "dice_rl_cartpole":
        from occupancy_ratio_benchmark.dice_rl_repro import (  # noqa: PLC0415
            make_dice_rl_reproduction_dataset,
        )

        collection = _mapping(resolved_config, "collection")
        behavior_alpha = float(_required(cell, "behavior_alpha"))
        dataset = make_dice_rl_reproduction_dataset(
            setting="dice_rl_cartpole",
            dataset_variant=f"alpha={behavior_alpha:g}",
            gamma=gamma,
            sample_size=sample_size,
            seed=audit_seed,
            dice_rl_repo_path=paths.dice_rl,
            asset_cache_dir=paths.asset_cache,
            install_assets=bool(paths.install_assets),
            num_trajectories=_positive_int(
                _required(collection, "trajectories"), "collection.trajectories"
            ),
            max_trajectory_length=_positive_int(
                _required(collection, "horizon"), "collection.horizon"
            ),
            target_value_rollouts=1,
            target_occupancy_trajectories_per_pool=0,
            collection_batch_size=20,
            include_target_evaluation=False,
        )
    else:
        raise ValueError(f"unsupported calibration benchmark_family {family!r}")

    validate_normalized_dataset(dataset)
    source_groups, initial_groups = independent_audit_group_ids(dataset)
    dataset.metadata = {
        **dataset.metadata,
        "independent_behavior_audit": True,
        "independent_behavior_audit_seed": int(audit_seed),
        "independent_behavior_audit_role": "external_c_a_b_evaluation_only",
        "calibration_benchmark_family": family,
    }
    return CalibrationDatasetBundle(
        dataset=dataset,
        source_groups=source_groups,
        initial_groups=initial_groups,
    )


def independent_audit_group_ids(dataset: BenchmarkDataset) -> tuple[Array, Array]:
    """Return group ids for a trajectory-level external audit split."""

    if dataset.episode_ids is not None:
        source = np.asarray(
            [f"audit-episode:{_scalar_id(value)}" for value in np.asarray(dataset.episode_ids).reshape(-1)],
            dtype=np.str_,
        )
        if dataset.initial_episode_ids is None:
            raise ValueError("trajectory audit data require initial_episode_ids")
        initial = np.asarray(
            [f"audit-episode:{_scalar_id(value)}" for value in np.asarray(dataset.initial_episode_ids).reshape(-1)],
            dtype=np.str_,
        )
        return source, initial
    return (
        np.asarray([f"audit-transition:{index}" for index in range(dataset.n)], dtype=np.str_),
        np.asarray(
            [f"audit-initial:{index}" for index in range(dataset.initial_states.shape[0])],
            dtype=np.str_,
        ),
    )


def validate_train_audit_independence(
    train: CalibrationDatasetBundle,
    audit: CalibrationDatasetBundle,
) -> None:
    """Fail closed when the external behavior audit can overlap fitting data."""

    if audit.dataset.metadata.get("independent_behavior_audit") is not True:
        raise ValueError("audit bundle is not marked as independent")
    train_family = str(train.dataset.metadata.get("calibration_benchmark_family", ""))
    audit_family = str(audit.dataset.metadata.get("calibration_benchmark_family", ""))
    if train_family != audit_family:
        raise ValueError("training and audit benchmark families differ")
    if train_family == "d4rl_matched":
        train_ids = set(np.asarray(train.dataset.initial_episode_ids).reshape(-1).tolist())
        audit_ids = set(np.asarray(audit.dataset.initial_episode_ids).reshape(-1).tolist())
        if train_ids & audit_ids:
            raise ValueError("D4RL training and audit raw episodes overlap")
        train_digest = train.dataset.metadata.get("behavior_episode_partition_sha256")
        audit_digest = audit.dataset.metadata.get("behavior_episode_partition_sha256")
        if not train_digest or train_digest != audit_digest:
            raise ValueError("D4RL training and audit partition provenance differs")
    else:
        train_seed = int(train.dataset.metadata.get("sample_seed", train.dataset.seed))
        audit_seed = int(audit.dataset.metadata["independent_behavior_audit_seed"])
        if train_seed == audit_seed:
            raise ValueError("training and audit sampling seeds must differ")


def _independent_audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(config, "evaluation")
    calibration_error = _mapping(evaluation, "calibration_error")
    value = calibration_error.get("independent_behavior_audit", {})
    if not isinstance(value, Mapping):
        raise ValueError("evaluation.calibration_error.independent_behavior_audit must be an object")
    return {
        "enabled": value.get("enabled") is True,
        "partition_seed": _integer(value.get("partition_seed", 20260813), "partition_seed"),
        "sample_seed_salt": _integer(value.get("sample_seed_salt", 914273), "sample_seed_salt"),
        "d4rl_raw_episode_fraction": float(value.get("d4rl_raw_episode_fraction", 0.2)),
    }


def _audit_seed(
    *,
    config_id: str,
    cell_id: str,
    sample_size: int,
    gamma: float,
    seed: int,
    salt: int,
) -> int:
    payload = (
        f"independent-behavior-audit|{config_id}|{cell_id}|{sample_size}|"
        f"{gamma:.12g}|{seed}|{salt}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def calibration_group_ids(dataset: BenchmarkDataset) -> tuple[Array, Array]:
    """Create stable episode groups, grouping exact duplicate transitions."""

    if dataset.episode_ids is not None:
        source = np.asarray(
            [f"episode:{_scalar_id(value)}" for value in np.asarray(dataset.episode_ids).reshape(-1)],
            dtype=np.str_,
        )
        if dataset.initial_episode_ids is None:
            initial = np.asarray(
                [f"initial:{index}" for index in range(dataset.initial_states.shape[0])],
                dtype=np.str_,
            )
        else:
            initial = np.asarray(
                [
                    f"episode:{_scalar_id(value)}"
                    for value in np.asarray(dataset.initial_episode_ids).reshape(-1)
                ],
                dtype=np.str_,
            )
        return source, initial

    source = np.asarray(
        [
            f"transition:{digest}"
            for digest in _row_digests(
                dataset.states,
                dataset.actions,
                dataset.next_states,
            )
        ],
        dtype=np.str_,
    )
    initial = np.asarray(
        [f"initial:{index}" for index in range(dataset.initial_states.shape[0])],
        dtype=np.str_,
    )
    return source, initial


def validate_normalized_dataset(dataset: BenchmarkDataset) -> None:
    if dataset.next_retention is not None and not np.allclose(dataset.next_retention, 1.0):
        raise ValueError("coverage-stopped next retention is forbidden")
    if dataset.initial_retention is not None and not np.allclose(dataset.initial_retention, 1.0):
        raise ValueError("coverage-stopped initial retention is forbidden")
    if bool(float(dataset.metadata.get("coverage_stopped_target", 0.0))):
        raise ValueError("coverage-stopped datasets are forbidden")
    if dataset.target_occupancy_mass is not None and not np.isclose(
        float(dataset.target_occupancy_mass), 1.0
    ):
        raise ValueError("target occupancy mass must be one for normalized calibration")


def _validate_target_occupancy_pools(
    dataset: BenchmarkDataset,
    *,
    trajectories_per_pool: int,
) -> None:
    per_pool = _nonnegative_int(
        trajectories_per_pool,
        "target_occupancy_trajectories_per_pool",
    )
    metadata_count = _nonnegative_int(
        dataset.metadata.get("target_occupancy_trajectories_per_pool", 0),
        "metadata.target_occupancy_trajectories_per_pool",
    )
    if metadata_count != per_pool:
        raise ValueError(
            "target occupancy telemetry does not match the requested trajectories per pool"
        )
    status = str(dataset.metadata.get("target_occupancy_status", ""))
    target_arrays = (
        dataset.target_occupancy_states,
        dataset.target_occupancy_actions,
        dataset.target_occupancy_episode_ids,
        dataset.target_occupancy_pool_ids,
    )
    if per_pool == 0:
        if status != "disabled" or any(value is not None for value in target_arrays):
            raise ValueError("disabled target occupancy must have disabled status and no pool arrays")
        return
    if status not in {"mc_rollout_ok", "cache:mc_rollout_ok"}:
        raise ValueError(f"target occupancy cache/rollout is not usable: {status or 'missing status'}")
    if any(value is None for value in target_arrays):
        raise ValueError("enabled target occupancy requires complete pool arrays")

    states = np.asarray(dataset.target_occupancy_states)
    actions = np.asarray(dataset.target_occupancy_actions)
    episode_ids = np.asarray(dataset.target_occupancy_episode_ids).reshape(-1)
    pool_ids = np.asarray(dataset.target_occupancy_pool_ids).reshape(-1)
    expected = 2 * per_pool
    if states.shape[0] != expected or actions.shape[0] != expected:
        raise ValueError(f"target occupancy must contain exactly {expected} pooled trajectories")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("target occupancy pool arrays must be finite")
    if np.unique(episode_ids).size != expected:
        raise ValueError("target occupancy episode ids must identify independent trajectories")
    pool_values, pool_counts = np.unique(pool_ids, return_counts=True)
    if not np.array_equal(pool_values, np.asarray([0, 1])) or not np.array_equal(
        pool_counts,
        np.asarray([per_pool, per_pool]),
    ):
        raise ValueError("target occupancy must contain two equal, labeled pools")


def write_dataset_bundle(path: str | Path, bundle: CalibrationDatasetBundle) -> dict[str, Any]:
    """Atomically write arrays plus a digest-checked JSON sidecar."""

    json_path, npz_path = _paired_paths(path)
    arrays: dict[str, Array] = {
        "source_groups": np.asarray(bundle.source_groups),
        "initial_groups": np.asarray(bundle.initial_groups),
    }
    present_fields = []
    for name in _ARRAY_FIELDS:
        value = getattr(bundle.dataset, name)
        if value is not None:
            arrays[name] = np.asarray(value)
            present_fields.append(name)
    _atomic_write_npz(npz_path, arrays)
    npz_sha = _file_sha256(npz_path)
    payload = {
        "schema_version": 1,
        "array_file": npz_path.name,
        "array_sha256": npz_sha,
        "array_fields": present_fields,
        "setting": bundle.dataset.setting,
        "gamma": float(bundle.dataset.gamma),
        "seed": int(bundle.dataset.seed),
        "sample_size": int(bundle.dataset.sample_size),
        "target_occupancy_mass": bundle.dataset.target_occupancy_mass,
        "target_policy_value": bundle.dataset.target_policy_value,
        "target_policy_value_se": bundle.dataset.target_policy_value_se,
        "target_policy_value_kind": bundle.dataset.target_policy_value_kind,
        "metadata": _json_safe(bundle.dataset.metadata),
        "truth_stages": _json_safe(bundle.truth_stages),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    _atomic_write_json(json_path, payload)
    return payload


def read_dataset_bundle(path: str | Path) -> CalibrationDatasetBundle:
    json_path, npz_path = _paired_paths(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    expected_payload = payload.get("payload_sha256")
    without_digest = dict(payload)
    without_digest.pop("payload_sha256", None)
    if expected_payload != _payload_sha256(without_digest):
        raise ValueError(f"dataset metadata digest mismatch: {json_path}")
    if payload.get("array_file") != npz_path.name:
        raise ValueError("dataset array filename mismatch")
    if payload.get("array_sha256") != _file_sha256(npz_path):
        raise ValueError(f"dataset array digest mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    present = set(payload.get("array_fields", []))
    kwargs = {name: arrays[name] if name in present else None for name in _ARRAY_FIELDS}
    dataset = BenchmarkDataset(
        setting=str(payload["setting"]),
        gamma=float(payload["gamma"]),
        seed=int(payload["seed"]),
        sample_size=int(payload["sample_size"]),
        target_occupancy_mass=_optional_float(payload.get("target_occupancy_mass")),
        target_policy_value=_optional_float(payload.get("target_policy_value")),
        target_policy_value_se=_optional_float(payload.get("target_policy_value_se")),
        target_policy_value_kind=str(payload.get("target_policy_value_kind", "")),
        metadata=dict(payload.get("metadata", {})),
        **kwargs,
    )
    validate_normalized_dataset(dataset)
    return CalibrationDatasetBundle(
        dataset=dataset,
        source_groups=np.asarray(arrays["source_groups"]),
        initial_groups=np.asarray(arrays["initial_groups"]),
        truth_stages=tuple(payload.get("truth_stages", [])),
    )


def _adaptive_truth_dataset(
    build: Callable[[int], BenchmarkDataset],
    truth: Mapping[str, Any],
) -> tuple[BenchmarkDataset, tuple[dict[str, Any], ...]]:
    start = _positive_int(_required(truth, "rollout_start"), "truth.rollout_start")
    maximum = _positive_int(_required(truth, "rollout_maximum"), "truth.rollout_maximum")
    growth = _positive_int(_required(truth, "rollout_growth"), "truth.rollout_growth")
    if start > maximum or growth < 2:
        raise ValueError("adaptive truth requires start<=maximum and growth>=2")
    stages = []
    rollouts = start
    final: BenchmarkDataset | None = None
    while True:
        final = build(rollouts)
        if final.target_policy_value is None or final.target_policy_value_se is None:
            raise ValueError("target rollout builder did not provide finite value truth and SE")
        mean = float(final.target_policy_value)
        se = float(final.target_policy_value_se)
        sd = se * np.sqrt(rollouts)
        half_width = 1.959963984540054 * se
        target = 0.02 * max(abs(mean), sd, 1.0)
        precision_met = bool(half_width <= target)
        stages.append(
            {
                "rollouts": int(rollouts),
                "mean": mean,
                "standard_error": se,
                "standard_deviation": float(sd),
                "ci95_half_width": float(half_width),
                "target_half_width": float(target),
                "precision_met": precision_met,
            }
        )
        if precision_met or rollouts >= maximum:
            break
        rollouts = min(maximum, rollouts * growth)
    assert final is not None
    final.metadata = {
        **final.metadata,
        "target_truth_precision_met": stages[-1]["precision_met"],
        "target_truth_ci95_half_width": stages[-1]["ci95_half_width"],
        "target_truth_target_half_width": stages[-1]["target_half_width"],
        "target_truth_adaptive_stages": stages,
    }
    return final, tuple(stages)


def _row_digests(*arrays: Array) -> list[str]:
    matrices = []
    n = None
    for value in arrays:
        matrix = np.ascontiguousarray(np.asarray(value).reshape(np.asarray(value).shape[0], -1))
        if n is None:
            n = matrix.shape[0]
        elif matrix.shape[0] != n:
            raise ValueError("duplicate-transition grouping arrays have unequal rows")
        matrices.append(matrix)
    assert n is not None
    output = []
    for index in range(n):
        digest = hashlib.blake2b(digest_size=16, person=b"occ-cal-group")
        for matrix in matrices:
            row = np.ascontiguousarray(matrix[index])
            digest.update(str(row.dtype).encode("ascii") + b"\0")
            digest.update(str(row.shape).encode("ascii") + b"\0")
            digest.update(row.tobytes())
        output.append(digest.hexdigest())
    return output


def _paired_paths(path: str | Path) -> tuple[Path, Path]:
    raw = Path(path)
    json_path = raw if raw.suffix == ".json" else raw.with_suffix(".json")
    return json_path, json_path.with_suffix(".npz")


def _atomic_write_npz(path: Path, arrays: Mapping[str, Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _scalar_id(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, bool)):
        return str(value)
    if isinstance(value, float) and np.isfinite(value):
        return value.hex()
    raise ValueError("episode ids must be scalar strings or finite numbers")


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _integer(value: object, name: str) -> int:
    result = int(value)
    if isinstance(value, bool) or float(value) != result:
        raise ValueError(f"{name} must be an integer")
    return result


def _positive_int(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


__all__: Sequence[str] = (
    "CalibrationDatasetBundle",
    "DatasetPaths",
    "SUPPORTED_FAMILIES",
    "build_calibration_audit_dataset",
    "build_calibration_dataset",
    "calibration_group_ids",
    "independent_audit_group_ids",
    "read_dataset_bundle",
    "validate_train_audit_independence",
    "validate_normalized_dataset",
    "write_dataset_bundle",
)
