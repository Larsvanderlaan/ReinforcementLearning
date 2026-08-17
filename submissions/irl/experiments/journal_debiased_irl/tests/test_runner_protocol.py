from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import fields
from types import SimpleNamespace

import pytest

from assemble_paper_results import (
    PAPER_REPETITIONS,
    _require_config,
    _validate_main_rows,
    _wilson_interval,
)
from data_fusion_simulation import (
    DataFusionRunResult,
    FrozenOutcomeRegression,
    summarize_data_fusion_results,
)
from jrssb_simulation import (
    SingleRunResult,
    run_monte_carlo,
    save_results_csv,
    save_summary_csv,
    summarize_results,
)
from run_jrssb_simulation import (
    _apply_frozen_fore_config,
    _configuration_identity,
    _data_fusion_cell_is_complete,
    _data_fusion_cell_is_compatible,
    _load_data_fusion_cell,
    _load_or_fit_outcome_regression,
    _load_single_run_cell,
    _main_replication_seeds,
    _main_cell_is_complete,
    _main_cell_is_compatible,
    _ensure_cell_identity,
    write_data_fusion_latex_table,
    write_main_inference_latex_table,
)


def _single_run_row(*, n: int, seed: int) -> SingleRunResult:
    values = {}
    for field in fields(SingleRunResult):
        if field.name == "example_id":
            values[field.name] = "1a"
        elif field.name == "ratio_mode":
            values[field.name] = "neural-fore"
        elif field.name == "nuisance_method":
            values[field.name] = "test"
        elif field.name == "n":
            values[field.name] = n
        elif field.name == "seed":
            values[field.name] = seed
        else:
            values[field.name] = 0.0
    return SingleRunResult(**values)


def _data_fusion_row(*, n: int, seed: int) -> DataFusionRunResult:
    values = {}
    for field in fields(DataFusionRunResult):
        if field.name == "n":
            values[field.name] = n
        elif field.name == "seed":
            values[field.name] = seed
        elif field.name == "outcome_sample_size":
            values[field.name] = 1_000_000
        else:
            values[field.name] = 0.0
    return DataFusionRunResult(**values)


def test_frozen_fore_config_is_truth_blind_and_overrides_only_base_settings(tmp_path):
    path = tmp_path / "frozen_fore_config.json"
    path.write_text(
        json.dumps(
            {
                "uses_oracle_truth": False,
                "selected": {
                    "hidden_dims": [128, 128],
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-2,
                },
            }
        )
    )
    args = Namespace(
        fore_frozen_config=path,
        fore_hidden_sizes=[64, 64],
        fore_learning_rate=1e-3,
        fore_weight_decay=1e-3,
        fore_iteration_budgets=[10, 30, 100],
    )
    _apply_frozen_fore_config(args)
    assert args.fore_hidden_sizes == [128, 128]
    assert args.fore_learning_rate == pytest.approx(3e-4)
    assert args.fore_weight_decay == pytest.approx(1e-2)
    assert args.fore_iteration_budgets == [10, 30, 100]

    path.write_text(json.dumps({"uses_oracle_truth": True, "selected": {}}))
    with pytest.raises(ValueError, match="uses_oracle_truth=false"):
        _apply_frozen_fore_config(args)


def test_configuration_keyed_cell_loaders_require_the_exact_seed_set(tmp_path):
    n = 2500
    main_seed = 10_000 + 97 * n + 101
    main_path = tmp_path / "main.csv"
    save_results_csv([_single_run_row(n=n, seed=main_seed)], main_path)
    main_rows = _load_single_run_cell(main_path)
    assert _main_cell_is_complete(
        main_rows,
        repetitions=1,
        example_id="1a",
        n=n,
        ratio_mode="neural-fore",
    )
    assert not _main_cell_is_complete(
        main_rows,
        repetitions=2,
        example_id="1a",
        n=n,
        ratio_mode="neural-fore",
    )
    assert _main_cell_is_compatible(
        main_rows,
        repetitions=2,
        example_id="1a",
        n=n,
        ratio_mode="neural-fore",
    )

    fusion_seed = 404 + 97 * n
    fusion_path = tmp_path / "fusion.csv"
    save_summary_csv(
        [_data_fusion_row(n=n, seed=fusion_seed).as_dict()], fusion_path
    )
    fusion_rows = _load_data_fusion_cell(fusion_path)
    assert _data_fusion_cell_is_complete(
        fusion_rows, repetitions=1, n=n, seed=404
    )
    assert _data_fusion_cell_is_compatible(
        fusion_rows, repetitions=2, n=n, seed=404
    )


def test_replication_subset_and_checkpoint_callback_use_deterministic_seeds(
    monkeypatch,
):
    import jrssb_simulation as simulation

    def fake_replication(*, oracle, n, seed, example_id, ratio_mode):
        del oracle
        row = _single_run_row(n=n, seed=seed)
        row.example_id = example_id
        row.ratio_mode = ratio_mode
        return row

    monkeypatch.setattr(simulation, "run_single_replication_safe", fake_replication)
    oracle = SimpleNamespace(
        config=SimpleNamespace(mc_sample_sizes=(2500,), mc_repetitions=3)
    )
    checkpointed = []
    rows = run_monte_carlo(
        oracle,
        sample_sizes=(2500,),
        repetitions=3,
        example_ids=("1a",),
        ratio_mode="neural-fore",
        jobs=1,
        replication_indices=(2, 0),
        on_result=checkpointed.append,
    )
    expected = [
        10_000 * (index + 1) + 97 * 2500 + 101
        for index in (2, 0)
    ]
    assert [row.seed for row in rows] == expected
    assert [row.seed for row in checkpointed] == expected


def test_ratio_estimators_use_common_data_seeds(monkeypatch):
    import jrssb_simulation as simulation

    def fake_replication(*, oracle, n, seed, example_id, ratio_mode):
        del oracle
        row = _single_run_row(n=n, seed=seed)
        row.example_id = example_id
        row.ratio_mode = ratio_mode
        return row

    monkeypatch.setattr(simulation, "run_single_replication_safe", fake_replication)
    oracle = SimpleNamespace(
        config=SimpleNamespace(mc_sample_sizes=(2500,), mc_repetitions=2)
    )
    neural = run_monte_carlo(
        oracle, ratio_mode="neural-fore", jobs=1, seed_offset=17
    )
    exact = run_monte_carlo(
        oracle, ratio_mode="oracle-adaptive", jobs=1, seed_offset=17
    )
    assert [row.seed for row in neural] == [row.seed for row in exact]


def test_cached_outcome_regression_rejects_stale_generation_metadata(tmp_path):
    cache_path = tmp_path / "frozen.pkl"
    FrozenOutcomeRegression(
        model=object(),
        sample_size=1_000_000,
        seed=91_001,
        outcome_noise_sd=0.5,
        g_rmse=0.0,
        estimand_shift=0.0,
    ).save(cache_path)
    args = Namespace(
        fusion_g_cache=cache_path,
        outcome_sample_size=1_000_000,
        outcome_regression_seed=91_002,
        outcome_noise_sd=0.5,
    )
    with pytest.raises(ValueError, match="metadata does not match"):
        _load_or_fit_outcome_regression(object(), tmp_path, args)


def test_cached_outcome_regression_rejects_stale_normalization_policy(tmp_path):
    cache_path = tmp_path / "frozen.pkl"
    FrozenOutcomeRegression(
        model=object(),
        sample_size=1_000_000,
        seed=91_001,
        outcome_noise_sd=0.5,
        g_rmse=0.0,
        estimand_shift=0.0,
        normalization_policy_mode="behavior",
    ).save(cache_path)
    args = Namespace(
        fusion_g_cache=cache_path,
        outcome_sample_size=1_000_000,
        outcome_regression_seed=91_001,
        outcome_noise_sd=0.5,
    )
    with pytest.raises(ValueError, match="expected 'known-uniform'"):
        _load_or_fit_outcome_regression(object(), tmp_path, args)


def test_summaries_report_failures_but_exclude_them_from_performance_metrics():
    success = _single_run_row(n=2500, seed=1)
    success.plugin_estimate = 1.0
    success.if_estimate = 2.0
    success.plugin_error = 1.0
    success.if_error = 2.0
    success.estimated_se = 0.5
    success.covered = 1.0
    failure = _single_run_row(n=2500, seed=2)
    failure.ratio_failure = 1.0
    failure.plugin_estimate = float("nan")
    failure.if_estimate = float("nan")
    failure.plugin_error = float("nan")
    failure.if_error = float("nan")
    failure.failure_message = "FloatingPointError: nonfinite"
    summary = summarize_results([success, failure])[0]
    assert summary["repetitions"] == 2
    assert summary["successful_repetitions"] == 1
    assert summary["plugin_bias"] == pytest.approx(1.0)
    assert summary["if_bias"] == pytest.approx(2.0)
    assert summary["ratio_failure_rate"] == pytest.approx(0.5)

    fusion_success = _data_fusion_row(n=2500, seed=1)
    fusion_success.plugin_estimate = 1.0
    fusion_success.if_estimate = 2.0
    fusion_success.plugin_error = 1.0
    fusion_success.if_error = 2.0
    fusion_failure = _data_fusion_row(n=2500, seed=2)
    fusion_failure.ratio_failure = 1.0
    fusion_failure.plugin_estimate = float("nan")
    fusion_failure.if_estimate = float("nan")
    fusion_failure.plugin_error = float("nan")
    fusion_failure.if_error = float("nan")
    fusion_summary = summarize_data_fusion_results(
        [fusion_success, fusion_failure]
    )[0]
    assert fusion_summary["repetitions"] == 2
    assert fusion_summary["successful_repetitions"] == 1
    assert fusion_summary["plugin_bias"] == pytest.approx(1.0)
    assert fusion_summary["ratio_failure_rate"] == pytest.approx(0.5)


def test_reader_tables_report_monte_carlo_sd_for_both_estimators(tmp_path):
    shared = {
        "plugin_bias": 0.1,
        "plugin_sd": 0.2,
        "plugin_rmse": 0.3,
        "if_bias": 0.01,
        "if_sd": 0.12,
        "if_rmse": 0.13,
        "avg_estimated_se": 0.11,
        "coverage_95": 0.95,
        "avg_ci_length": 0.44,
        "ratio_failure_rate": 0.0,
    }
    main_path = tmp_path / "main.tex"
    write_main_inference_latex_table(
        [shared | {"example_id": "1a", "n": 2500}], main_path
    )
    main_text = main_path.read_text()
    assert "MC SD" in main_text
    assert "Plug-in" in main_text and "D-IRL" in main_text
    assert "0.200" in main_text and "0.120" in main_text

    fusion_path = tmp_path / "fusion.tex"
    write_data_fusion_latex_table([shared | {"n": 2500}], fusion_path)
    fusion_text = fusion_path.read_text()
    assert "MC SD" in fusion_text
    assert "0.200" in fusion_text and "0.120" in fusion_text


def test_source_aware_identity_and_immutable_checkpoint_sidecar(tmp_path):
    args = Namespace(
        mode="monte-carlo",
        examples=["1a"],
        sample_sizes=[2500],
        repetitions=300,
        output_dir=tmp_path,
        jobs=1,
        resume=True,
        fore_frozen_config=None,
        fusion_g_cache=None,
        fusion_pilot_manifest=None,
    )
    identity = _configuration_identity(args, "main-inference")
    assert identity["result_source_sha256"]
    assert len(identity["configuration_sha256"]) == 64
    sidecar = tmp_path / "cell.identity.json"
    _ensure_cell_identity(
        sidecar,
        configuration_identity=identity,
        cell={"example_id": "1a", "n": 2500},
        checkpoint_exists=False,
    )
    _ensure_cell_identity(
        sidecar,
        configuration_identity=identity,
        cell={"example_id": "1a", "n": 2500},
        checkpoint_exists=True,
    )
    with pytest.raises(RuntimeError, match="mismatched immutable identity"):
        _ensure_cell_identity(
            sidecar,
            configuration_identity=identity,
            cell={"example_id": "1a", "n": 5000},
            checkpoint_exists=True,
        )


def test_main_cli_seed_changes_replication_seed_set():
    common = dict(repetitions=3, example_id="1a", n=2500)
    first = _main_replication_seeds(**common, effective_seed_offset=404)
    second = _main_replication_seeds(**common, effective_seed_offset=405)
    assert first != second
    assert all(right - left == 1 for left, right in zip(first, second))


def test_paper_gate_validates_raw_rows_and_is_fixed_at_300():
    assert PAPER_REPETITIONS == 300
    n = 2500
    seeds = _main_replication_seeds(
        repetitions=2,
        example_id="1a",
        n=n,
        effective_seed_offset=404,
    )
    rows = [_single_run_row(n=n, seed=seed) for seed in seeds]
    for row in rows:
        for field in (
            "estimated_se",
            "weight_ess",
            "fore_fit_seconds",
            "fore_training_rows",
            "iid_estimated_se",
            "ci_critical_value",
        ):
            setattr(row, field, 1.0)
        row.crossfit_fold_count = 5.0
        row.fore_selected_iterations = 30.0
        row.fore_normalized_mass = 1.0
        row.fore_selected_iterations_by_fold = json.dumps([30] * 5)
        row.fore_apbv_scores_by_fold = json.dumps([0.1] * 5)
        row.failure_message = ""
    _validate_main_rows(
        rows,
        example_id="1a",
        n=n,
        seed=404,
        seed_offset=0,
        repetitions=2,
    )
    rows[0].estimated_se = float("nan")
    with pytest.raises(ValueError, match="Nonfinite required fields"):
        _validate_main_rows(
            rows,
            example_id="1a",
            n=n,
            seed=404,
            seed_offset=0,
            repetitions=2,
        )


def test_legacy_or_oracle_config_is_rejected_and_coverage_is_diagnostic():
    expected = {
        "mode": "monte-carlo",
        "ratio_mode": "neural-fore",
        "crossfit_folds": 5,
    }
    with pytest.raises(ValueError, match="Locked configuration mismatch"):
        _require_config(
            {"mode": "monte-carlo", "ratio_mode": "oracle-adaptive", "crossfit_folds": 2},
            expected,
            Namespace(name="legacy-run"),
        )
    lower, upper = _wilson_interval(275, 300)
    assert lower < upper < 0.95
