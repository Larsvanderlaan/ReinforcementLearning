from __future__ import annotations

import json
import time

import numpy as np
import pytest

from occupancy_ratio_benchmark.calibration_artifacts import (
    ArtifactConflictError,
    artifact_is_ready,
    quarantine_stale_state,
    read_unit_arrays,
    write_unit_artifact,
    write_unit_state,
)


def _manifest():
    return {
        "run_id": "occ-cal-test",
        "manifest_payload_sha256": "a" * 64,
        "provenance": {"fingerprint": "b" * 64},
    }


def _unit():
    return {
        "unit_id": "fold-test",
        "expected_artifact": "fold_units/fold-test.json",
        "identity": {"kind": "cross_calibration_fold", "config_sha256": "c" * 64},
    }


def test_atomic_unit_round_trip_and_tamper_detection(tmp_path) -> None:
    write_unit_artifact(
        run_root=tmp_path,
        manifest=_manifest(),
        unit=_unit(),
        arrays={"source_q": np.arange(5, dtype=np.float64)},
        metadata={"metric": float("inf")},
        attempt=0,
        started_at=time.time(),
    )
    assert artifact_is_ready(run_root=tmp_path, manifest=_manifest(), unit=_unit())
    arrays = read_unit_arrays(run_root=tmp_path, manifest=_manifest(), unit=_unit())
    np.testing.assert_array_equal(arrays["source_q"], np.arange(5))

    path = tmp_path / "fold_units" / "fold-test.json"
    payload = json.loads(path.read_text())
    payload["metadata"]["metric"] = 3.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactConflictError, match="metadata digest"):
        artifact_is_ready(run_root=tmp_path, manifest=_manifest(), unit=_unit())


def test_stale_state_is_quarantined_not_deleted(tmp_path) -> None:
    write_unit_state(
        run_root=tmp_path,
        manifest=_manifest(),
        unit=_unit(),
        status="running",
        attempt=0,
        started_at=1.0,
    )
    state = tmp_path / "unit_state" / "fold-test.json"
    payload = json.loads(state.read_text())
    payload["heartbeat_at_unix"] = 1.0
    state.write_text(json.dumps(payload))
    quarantined = quarantine_stale_state(
        run_root=tmp_path,
        unit_id="fold-test",
        stale_after_sec=10.0,
        now=100.0,
    )
    assert quarantined is not None and quarantined.exists()
    assert not state.exists()
