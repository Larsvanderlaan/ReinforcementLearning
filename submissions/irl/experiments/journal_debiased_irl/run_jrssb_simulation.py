from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

EXPERIMENT_SCHEMA_VERSION = 3
PAPER_PROTOCOL_ID = "jasa-neural-fore-v2-corrected"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jrssb_simulation import (
    JRSSBConfig,
    JRSSBOracle,
    SingleRunResult,
    run_bias_acceptance,
    run_example1a_policy_audit,
    run_example1b_policy_audit,
    run_example2_method_selection,
    run_example2_nuisance_audit,
    run_example2_paper_comparison,
    run_example2_semi_oracle_audit,
    run_example2_shakedown,
    run_example2_smoke_check,
    run_monte_carlo,
    run_single_replication,
    run_validation_suite,
    fit_behavior_policy_example1b,
    fit_soft_value_from_reward,
    fold_splits,
    make_policy_adapter,
    save_json,
    save_results_csv,
    save_summary_csv,
    summarize_results,
)
from fore_ratio import (
    APBVSelectionResult,
    FORECandidateConfig,
    FOREFitOptions,
    aggregate_pilot_selections,
    deterministic_three_way_split,
    fit_selected_fore_ratio,
    fit_selected_signed_fore_ratio,
    paper_early_stopping_candidates,
    paper_pilot_candidates,
    selection_manifest,
)
from data_fusion_simulation import (
    DataFusionRunResult,
    FrozenOutcomeRegression,
    data_fusion_readiness,
    fit_frozen_outcome_regression,
    run_data_fusion_monte_carlo,
    summarize_data_fusion_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JRSSB IRL debiasing simulation study.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rl_evaluation_suite_jrssb_simulation"))
    parser.add_argument(
        "--mode",
        choices=[
            "checks",
            "bias-acceptance",
            "pilot",
            "fore-smoke",
            "fore-selection-pilot",
            "monte-carlo",
            "example1a-policy-audit",
            "example1b-policy-audit",
            "example2-method-pilot",
            "example2-nuisance-audit",
            "example2-paper-comparison",
            "example2-semi-oracle",
            "example2-smoke",
            "example2-shakedown",
            "data-fusion-smoke",
            "data-fusion-pilot",
            "data-fusion-confirmatory",
        ],
        default="checks",
    )
    parser.add_argument("--examples", nargs="*", default=["1a", "1b"])
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument(
        "--ratio-mode",
        choices=["neural-fore", "oracle", "oracle-adaptive", "coarse-estimated"],
        default="neural-fore",
    )
    parser.add_argument(
        "--allow-oracle-diagnostics",
        action="store_true",
        default=False,
        help="Permit explicitly labeled oracle behavior/adaptive-ratio diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument(
        "--replication-seed-offset",
        type=int,
        default=0,
        help="Shared data-seed offset; keep identical across estimator comparisons.",
    )
    parser.add_argument("--pilot-n", type=int, default=10_000)
    parser.add_argument("--checks-large-n", type=int, default=5_000)
    parser.add_argument("--main-grid-points", type=int, default=41)
    parser.add_argument("--coarse-grid-points", type=int, default=31)
    parser.add_argument(
        "--behavior-policy-design",
        choices=["auto", "quadratic-logit", "soft-mdp-stress"],
        default="auto",
    )
    parser.add_argument(
        "--state-jitter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mc-repetitions", type=int, default=None)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--crossfit-se-method",
        choices=["iid", "fold-cluster", "fold-cluster-max"],
        default="iid",
    )
    parser.add_argument(
        "--crossfit-ci-method",
        choices=["normal", "fold-t"],
        default="normal",
    )
    parser.add_argument("--nuisance-sample-mode", choices=["crossfit", "independent"], default="crossfit")
    parser.add_argument("--fore-hidden-sizes", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--fore-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fore-weight-decay", type=float, default=1e-3)
    parser.add_argument("--fore-iteration-budgets", nargs="+", type=int, default=[30, 100, 300])
    parser.add_argument("--fore-batch-size", type=int, default=512)
    parser.add_argument("--fore-optimizer-steps", type=int, default=5)
    parser.add_argument("--fore-target-action-draws", type=int, default=4)
    parser.add_argument("--fore-logit-clip", type=float, default=10.0)
    parser.add_argument("--fore-grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--fore-device", type=str, default="cpu")
    parser.add_argument(
        "--fore-max-training-rows",
        type=int,
        default=20_000,
        help=(
            "Deterministic per-fold neural ratio optimization budget; zero "
            "uses all outer-training rows."
        ),
    )
    parser.add_argument(
        "--fore-frozen-config",
        type=Path,
        default=None,
        help="Truth-blind frozen_fore_config.json produced by fore-selection-pilot.",
    )
    parser.add_argument("--expected-fore-sha256", type=str, default=None)
    parser.add_argument("--outcome-sample-size", type=int, default=1_000_000)
    parser.add_argument("--outcome-noise-sd", type=float, default=0.2)
    parser.add_argument("--outcome-regression-seed", type=int, default=91_001)
    parser.add_argument("--fusion-g-cache", type=Path, default=None)
    parser.add_argument("--expected-fusion-g-sha256", type=str, default=None)
    parser.add_argument("--fusion-pilot-manifest", type=Path, default=None)
    parser.add_argument(
        "--expected-fusion-pilot-manifest-sha256", type=str, default=None
    )
    parser.add_argument("--paper-protocol-id", type=str, default=None)
    parser.add_argument(
        "--data-fusion-policy-mode",
        choices=["known-logging", "sieve-logit", "oracle"],
        default="known-logging",
    )
    parser.add_argument(
        "--data-fusion-transition-mode",
        choices=["sieve", "oracle"],
        default="sieve",
    )
    parser.add_argument(
        "--data-fusion-g-mode",
        choices=["frozen", "oracle"],
        default="frozen",
    )
    parser.add_argument(
        "--data-fusion-ratio-mode",
        choices=["neural-fore", "oracle-adaptive"],
        default="neural-fore",
    )
    parser.add_argument("--data-fusion-target-gamma", type=float, default=0.80)
    parser.add_argument("--data-fusion-repeated-splits", type=int, default=1)
    parser.add_argument("--data-fusion-probability-floor", type=float, default=0.02)
    parser.add_argument("--fore-pilot-repetitions", type=int, default=10)
    parser.add_argument("--fusion-pilot-repetitions", type=int, default=20)
    parser.add_argument("--fusion-pilot-median-se", type=float, default=None)
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--fqe-iters", type=int, default=12)
    parser.add_argument("--fqe-epochs", type=int, default=8)
    parser.add_argument(
        "--example1a-nuisance-method",
        choices=["neural-fqe", "neural-main-oracle-bellman"],
        default="neural-main-oracle-bellman",
    )
    parser.add_argument(
        "--example1a-policy-estimator",
        choices=[
            "bc",
            "maxent",
            "sieve-logit",
            "coarse",
            "blend",
            "structural-linear",
            "oracle",
        ],
        default="sieve-logit",
    )
    parser.add_argument("--example1a-repeated-splits", type=int, default=1)
    parser.add_argument(
        "--main-sieve-mode",
        choices=["fixed-quadratic", "selected"],
        default="fixed-quadratic",
    )
    parser.add_argument("--main-sieve-degree", type=int, default=2)
    parser.add_argument("--main-sieve-c", type=float, default=10.0)
    parser.add_argument("--example1a-stabilized-lambda", type=float, default=0.60)
    parser.add_argument(
        "--example1b-nuisance-method",
        choices=["neural-fqe", "neural-main-oracle-bellman"],
        default="neural-main-oracle-bellman",
    )
    parser.add_argument(
        "--example1b-policy-estimator",
        choices=[
            "bc",
            "maxent",
            "sieve-logit",
            "structural",
            "structural-linear",
            "oracle",
        ],
        default="sieve-logit",
    )
    parser.add_argument("--example1b-repeated-splits", type=int, default=1)
    parser.add_argument("--example1b-structural-iters", type=int, default=300)
    parser.add_argument("--example1b-structural-bellman-iters", type=int, default=20)
    parser.add_argument("--example1b-structural-learning-rate", type=float, default=0.05)
    parser.add_argument("--example1b-structural-smoothness-penalty", type=float, default=0.5)
    parser.add_argument("--example1b-structural-reward-scale", type=float, default=4.0)
    parser.add_argument(
        "--example2-nuisance-method",
        choices=[
            "neural",
            "coarse-oracle-bellman",
            "neural-coarse-bellman",
            "neural-main-oracle-bellman",
            "oracle-q",
            "oracle-all",
        ],
        default="neural-main-oracle-bellman",
    )
    parser.add_argument(
        "--example2-policy-estimator",
        choices=["bc", "maxent", "structural-linear"],
        default="maxent",
    )
    parser.add_argument(
        "--example2-methods",
        nargs="*",
        default=["neural", "neural-main-oracle-bellman", "neural-coarse-bellman"],
    )
    parser.add_argument("--example2-semi-oracle-n", type=int, default=2500)
    parser.add_argument("--example2-repeated-splits", type=int, default=1)
    parser.add_argument("--example2-ci-critical-value", type=float, default=1.96)
    parser.add_argument("--shakedown-reps", type=int, default=20)
    parser.add_argument("--acceptance-audit-repetitions", type=int, default=10)
    parser.add_argument("--acceptance-large-n", type=int, default=50_000)
    parser.add_argument("--acceptance-large-repetitions", type=int, default=5)
    parser.add_argument("--acceptance-pilot-repetitions", type=int, default=20)
    parser.add_argument("--acceptance-full-repetitions", type=int, default=50)
    parser.add_argument("--acceptance-confirmation-repetitions", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete, configuration-keyed cell files.",
    )
    parser.add_argument("--no-oracle-cache", action="store_true", default=False)
    return parser.parse_args()


def maybe_plot(summary_rows: Sequence[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    example_ids = [
        example_id
        for example_id in ("1a", "1b", "2")
        if any(row["example_id"] == example_id for row in summary_rows)
    ]
    if not example_ids:
        return
    labels = {"1a": "Example 1a", "1b": "Example 1b", "2": "Example 2"}
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(example_ids),
        figsize=(5.3 * len(example_ids), 4),
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]
    for axis, example_id in zip(axes, example_ids):
        rows = [row for row in summary_rows if row["example_id"] == example_id]
        if not rows:
            continue
        methods = sorted({row.get("nuisance_method", "default") for row in rows})
        multi_method = len(methods) > 1
        for method in methods:
            method_rows = sorted((row for row in rows if row.get("nuisance_method", "default") == method), key=lambda row: row["n"])
            n_vals = [row["n"] for row in method_rows]
            suffix = f" ({method})" if multi_method else ""
            axis.plot(
                n_vals,
                [abs(row["plugin_bias"]) for row in method_rows],
                marker="o",
                label=f"|Bias| plugin{suffix}",
            )
            axis.plot(
                n_vals,
                [abs(row["if_bias"]) for row in method_rows],
                marker="o",
                linestyle="--",
                label=f"|Bias| IF{suffix}",
            )
            axis.plot(
                n_vals,
                [row["plugin_rmse"] for row in method_rows],
                marker="s",
                linestyle=":",
                label=f"RMSE plugin{suffix}",
            )
            axis.plot(
                n_vals,
                [row["if_rmse"] for row in method_rows],
                marker="s",
                linestyle="-.",
                label=f"RMSE IF{suffix}",
            )
        axis.set_title(labels[example_id])
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Error scale")
    fig.supxlabel("n")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "bias_rmse_panels.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(example_ids),
        figsize=(5.3 * len(example_ids), 4),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]
    for axis, example_id in zip(axes, example_ids):
        rows = [row for row in summary_rows if row["example_id"] == example_id]
        if not rows:
            continue
        methods = sorted({row.get("nuisance_method", "default") for row in rows})
        for method in methods:
            method_rows = sorted((row for row in rows if row.get("nuisance_method", "default") == method), key=lambda row: row["n"])
            n_vals = [row["n"] for row in method_rows]
            axis.plot(n_vals, [row["coverage_95"] for row in method_rows], marker="o", label=method)
        axis.axhline(0.95, color="#C44E52", linestyle="--", linewidth=1.0)
        axis.set_title(labels[example_id])
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("95% coverage")
    fig.supxlabel("n")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(output_dir / "coverage_panels.png", dpi=200)
    plt.close(fig)

    ex2_rows = [row for row in summary_rows if row["example_id"] == "2"]
    if ex2_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        methods = sorted({row.get("nuisance_method", "default") for row in ex2_rows})
        for method in methods:
            method_rows = sorted((row for row in ex2_rows if row.get("nuisance_method", "default") == method), key=lambda row: row["n"])
            n_vals = [row["n"] for row in method_rows]
            axes[0].plot(n_vals, [row["avg_pi0_action0_q01"] for row in method_rows], marker="o", label=method)
            axes[1].plot(n_vals, [row["avg_pi_ratio_q99"] for row in method_rows], marker="o", label=method)
            axes[2].plot(n_vals, [row["avg_nu_ratio_q99"] for row in method_rows], marker="o", label=method)
        axes[0].set_title("q01 of $\\hat\\pi(0|S)$")
        axes[1].set_title("q99 of $\\pi/\\hat\\pi$")
        axes[2].set_title("q99 of $\\nu/\\hat\\pi$")
        for axis in axes:
            axis.set_xlabel("n")
            axis.set_xscale("log")
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "example2_overlap_panel.png", dpi=200)
        plt.close(fig)


def maybe_plot_data_fusion(summary_rows: Sequence[dict], output_dir: Path) -> None:
    """Render the compact paper-facing data-fusion diagnostics."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not summary_rows:
        return
    rows = sorted(summary_rows, key=lambda row: row["n"])
    sample_sizes = [row["n"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(
        sample_sizes,
        [row["plugin_rmse"] for row in rows],
        marker="o",
        label="Plug-in",
    )
    axes[0].plot(
        sample_sizes,
        [row["if_rmse"] for row in rows],
        marker="o",
        label="Debiased",
    )
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)
    axes[1].plot(
        sample_sizes,
        [row["coverage_95"] for row in rows],
        marker="o",
    )
    axes[1].axhline(0.95, color="#C44E52", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("95% coverage")
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.set_xlabel("Transition sample size")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "data_fusion_performance.png", dpi=200)
    plt.close(fig)


def _latex_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_main_inference_latex_table(
    summary_rows: Sequence[dict], path: Path
) -> None:
    """Write the compact reader-facing estimated-ratio summary table."""
    rows = sorted(summary_rows, key=lambda row: (row["example_id"], row["n"]))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Estimated-ratio inference with neural FORE.}",
        r"\label{tab:fore-inference}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Design & $n$ & Estimator & Bias & MC SD & RMSE & Mean SE & Coverage & Width & Fail. \\",
        r"\midrule",
    ]
    for row in rows:
        design = str(row["example_id"])
        lines.append(
            " & ".join(
                [
                    design,
                    str(int(row["n"])),
                    "Plug-in",
                    _latex_number(row["plugin_bias"]),
                    _latex_number(row["plugin_sd"]),
                    _latex_number(row["plugin_rmse"]),
                    "--",
                    "--",
                    "--",
                    _latex_number(row["ratio_failure_rate"]),
                ]
            )
            + r" \\"
        )
        lines.append(
            " & ".join(
                [
                    "",
                    "",
                    "D-IRL",
                    _latex_number(row["if_bias"]),
                    _latex_number(row["if_sd"]),
                    _latex_number(row["if_rmse"]),
                    _latex_number(row["avg_estimated_se"]),
                    _latex_number(row["coverage_95"]),
                    _latex_number(row["avg_ci_length"]),
                    _latex_number(row["ratio_failure_rate"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_data_fusion_latex_table(
    summary_rows: Sequence[dict], path: Path
) -> None:
    """Write the compact reader-facing data-fusion summary table."""
    rows = sorted(summary_rows, key=lambda row: row["n"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Data-fusion estimation conditional on the frozen outcome source.}",
        r"\label{tab:data-fusion}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"$n$ & Estimator & Bias & MC SD & RMSE & Mean SE & Coverage & Width & Fail. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(int(row["n"])),
                    "Plug-in",
                    _latex_number(row["plugin_bias"]),
                    _latex_number(row["plugin_sd"]),
                    _latex_number(row["plugin_rmse"]),
                    "--",
                    "--",
                    "--",
                    _latex_number(row["ratio_failure_rate"]),
                ]
            )
            + r" \\"
        )
        lines.append(
            " & ".join(
                [
                    "",
                    "D-IRL",
                    _latex_number(row["if_bias"]),
                    _latex_number(row["if_sd"]),
                    _latex_number(row["if_rmse"]),
                    _latex_number(row["avg_estimated_se"]),
                    _latex_number(row["coverage_95"]),
                    _latex_number(row["avg_ci_length"]),
                    _latex_number(row["ratio_failure_rate"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_apbv_selection_behavior(
    decision: dict[str, object],
    serialized_events: Sequence[dict[str, object]],
    path: Path,
) -> None:
    """Plot truth-blind base-configuration and stopping-budget selections."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional outside paper runs
        return
    summary = sorted(
        decision["summary_rows"],
        key=lambda row: (
            sum(row["hidden_dims"]),
            row["learning_rate"],
            row["weight_decay"],
        ),
    )
    labels = [
        f"{'x'.join(map(str, row['hidden_dims']))}\n"
        f"lr={row['learning_rate']:g}, wd={row['weight_decay']:g}"
        for row in summary
    ]
    frequencies = [int(row["selection_count"]) for row in summary]
    components = ("ordinary", "signed-positive", "signed-negative")
    observed_budgets = {
        int(str(event["selection"]["candidate_ids"][int(event["selection"]["selected_index"])]).rsplit("_k", 1)[1])
        for event in serialized_events
        if event.get("selection") is not None
        and str(event.get("component")) in components
    }
    budgets = tuple(sorted(observed_budgets)) or (30, 100, 300)
    budget_counts = {component: {budget: 0 for budget in budgets} for component in components}
    for event in serialized_events:
        selection = event.get("selection")
        component = str(event.get("component"))
        if selection is None or component not in budget_counts:
            continue
        selected_index = int(selection["selected_index"])
        candidate_id = str(selection["candidate_ids"][selected_index])
        budget = int(candidate_id.rsplit("_k", 1)[1])
        budget_counts[component][budget] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(np.arange(len(summary)), frequencies, color="#315f8c")
    axes[0].set_xticks(np.arange(len(summary)), labels, rotation=35, ha="right")
    axes[0].set_ylabel("A-PBV selection count")
    axes[0].set_title("Frozen architecture/optimizer pilot")
    x = np.arange(len(budgets))
    width = 0.25
    for offset, component in enumerate(components):
        axes[1].bar(
            x + (offset - 1) * width,
            [budget_counts[component][budget] for budget in budgets],
            width,
            label=component.replace("signed-", "signed "),
        )
    axes[1].set_xticks(x, [str(budget) for budget in budgets])
    axes[1].set_xlabel("Selected outer iterations")
    axes[1].set_ylabel("Fold/component count")
    axes[1].set_title("Truth-blind early stopping")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_config(args: argparse.Namespace) -> JRSSBConfig:
    behavior_policy_design = resolved_behavior_policy_design(args)
    config = JRSSBConfig(
        main_grid_points=args.main_grid_points,
        coarse_grid_points=args.coarse_grid_points,
        behavior_policy_design=behavior_policy_design,
        state_jitter=args.state_jitter,
        bc_epochs=args.bc_epochs,
        fqe_iters=args.fqe_iters,
        fqe_epochs_per_iter=args.fqe_epochs,
        crossfit_folds=args.crossfit_folds,
        crossfit_se_method=args.crossfit_se_method,
        crossfit_ci_method=args.crossfit_ci_method,
        nuisance_sample_mode=args.nuisance_sample_mode,
        fore_hidden_sizes=tuple(args.fore_hidden_sizes),
        fore_learning_rate=args.fore_learning_rate,
        fore_weight_decay=args.fore_weight_decay,
        fore_iteration_budgets=tuple(args.fore_iteration_budgets),
        fore_batch_size=args.fore_batch_size,
        fore_optimizer_steps=args.fore_optimizer_steps,
        fore_target_action_draws=args.fore_target_action_draws,
        fore_logit_clip=args.fore_logit_clip,
        fore_grad_clip_norm=args.fore_grad_clip_norm,
        fore_device=args.fore_device,
        fore_max_training_rows=args.fore_max_training_rows,
        data_fusion_policy_mode=args.data_fusion_policy_mode,
        data_fusion_transition_mode=args.data_fusion_transition_mode,
        data_fusion_g_mode=args.data_fusion_g_mode,
        data_fusion_ratio_mode=args.data_fusion_ratio_mode,
        data_fusion_target_gamma=args.data_fusion_target_gamma,
        data_fusion_repeated_splits=args.data_fusion_repeated_splits,
        data_fusion_probability_floor=args.data_fusion_probability_floor,
        example1a_nuisance_method=args.example1a_nuisance_method,
        example1a_policy_estimator=args.example1a_policy_estimator,
        main_sieve_mode=args.main_sieve_mode,
        main_sieve_degree=args.main_sieve_degree,
        main_sieve_c=args.main_sieve_c,
        example1a_repeated_splits=args.example1a_repeated_splits,
        example1a_stabilized_lambda=args.example1a_stabilized_lambda,
        example1b_nuisance_method=args.example1b_nuisance_method,
        example1b_policy_estimator=args.example1b_policy_estimator,
        example1b_repeated_splits=args.example1b_repeated_splits,
        example1b_structural_iters=args.example1b_structural_iters,
        example1b_structural_bellman_iters=args.example1b_structural_bellman_iters,
        example1b_structural_learning_rate=args.example1b_structural_learning_rate,
        example1b_structural_smoothness_penalty=args.example1b_structural_smoothness_penalty,
        example1b_structural_reward_scale=args.example1b_structural_reward_scale,
        example2_nuisance_method=args.example2_nuisance_method,
        example2_policy_estimator=args.example2_policy_estimator,
        example2_repeated_splits=args.example2_repeated_splits,
        example2_ci_critical_value=args.example2_ci_critical_value,
        use_oracle_cache=not args.no_oracle_cache,
    )
    if args.mc_repetitions is not None:
        config.mc_repetitions = args.mc_repetitions
    return config


def resolved_behavior_policy_design(args: argparse.Namespace) -> str:
    """Return the concrete DGP behind the user-facing auto setting."""
    if args.behavior_policy_design != "auto":
        return str(args.behavior_policy_design)
    return (
        "soft-mdp-stress"
        if args.mode.startswith("data-fusion")
        else "quadratic-logit"
    )


def run_checks(oracle: JRSSBOracle, output_dir: Path, args: argparse.Namespace) -> None:
    diagnostics = run_validation_suite(
        oracle=oracle,
        pilot_seed=args.seed,
        pilot_n=args.pilot_n,
        oracle_sample_n=args.checks_large_n,
    )
    save_json({"mode": "checks", "diagnostics": diagnostics}, output_dir / "checks.json")


def run_bias_acceptance_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_bias_acceptance(
        base_config=config,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
        audit_repetitions=args.acceptance_audit_repetitions,
        large_n=args.acceptance_large_n,
        large_repetitions=args.acceptance_large_repetitions,
        pilot_repetitions=args.acceptance_pilot_repetitions,
        full_repetitions=args.acceptance_full_repetitions,
        confirmation_repetitions=args.acceptance_confirmation_repetitions,
    )
    save_summary_csv(study["candidate_diagnostics"]["example1a_audit"]["rows"], output_dir / "example1a_audit_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1a_audit"]["summary"], output_dir / "example1a_audit_summary.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_audit"]["rows"], output_dir / "example1b_audit_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_audit"]["summary"], output_dir / "example1b_audit_summary.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1a_large_sample"]["rows"], output_dir / "example1a_large_sample_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1a_large_sample"]["summary"], output_dir / "example1a_large_sample_summary.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_large_sample"]["rows"], output_dir / "example1b_large_sample_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_large_sample"]["summary"], output_dir / "example1b_large_sample_summary.csv")
    save_summary_csv(study["pilot_coverage"]["example1a"], output_dir / "pilot_coverage_example1a.csv")
    save_summary_csv(study["pilot_coverage"]["example1b"], output_dir / "pilot_coverage_example1b.csv")
    save_summary_csv(study["pilot_coverage"]["example2"], output_dir / "pilot_coverage_example2.csv")
    save_summary_csv(study["stage4_full_coverage"], output_dir / "full_coverage_stage4.csv")
    save_summary_csv(study["stage5_confirmation_coverage"], output_dir / "full_coverage_confirmation.csv")
    save_summary_csv(study["final_full_coverage"], output_dir / "full_coverage_final.csv")
    save_summary_csv(study["baseline_secondary_coverage"], output_dir / "baseline_secondary_coverage.csv")
    save_summary_csv(study["secondary_finite_sample_checks"], output_dir / "secondary_checks.csv")
    save_json(
        {
            "acceptance": study["acceptance"],
            "selections": study["selections"],
            "invariants": study["invariants"],
        },
        output_dir / "acceptance.json",
    )


def run_pilot(oracle: JRSSBOracle, output_dir: Path, args: argparse.Namespace) -> None:
    results = [
        run_single_replication(
            oracle=oracle,
            n=args.pilot_n,
            seed=args.seed + offset,
            example_id=example_id,
            ratio_mode=args.ratio_mode,
        )
        for offset, example_id in enumerate(args.examples)
    ]
    save_results_csv(results, output_dir / "pilot_results.csv")
    save_summary_csv(summarize_results(results), output_dir / "pilot_summary.csv")
    diagnostics = run_validation_suite(
        oracle=oracle,
        pilot_seed=args.seed,
        pilot_n=args.pilot_n,
        oracle_sample_n=args.checks_large_n,
    )
    save_json({"mode": "pilot", "diagnostics": diagnostics}, output_dir / "pilot_checks.json")


def _result_source_paths() -> dict[str, Path]:
    """Return code paths whose contents can change simulation rows."""
    source_dir = Path(__file__).resolve().parent
    repository = Path(__file__).resolve().parents[4]
    return {
        "run_jrssb_simulation.py": source_dir / "run_jrssb_simulation.py",
        "jrssb_simulation.py": source_dir / "jrssb_simulation.py",
        "fore_ratio.py": source_dir / "fore_ratio.py",
        "data_fusion_simulation.py": source_dir / "data_fusion_simulation.py",
        "conference_data_fusion.py": repository
        / "submissions/irl/experiments/conference_genpqr/repro/data_fusion.py",
        "conference_policy_estimation.py": repository
        / "submissions/irl/experiments/conference_genpqr/repro/policy_estimation.py",
        "conference_q_evaluation.py": repository
        / "submissions/irl/experiments/conference_genpqr/repro/q_evaluation.py",
        "occupancy_ratio_kl_fori.py": repository
        / "packages/occupancy-ratio/occupancy_ratio/_kl_fori_impl.py",
        "occupancy_ratio_neural.py": repository
        / "packages/occupancy-ratio/occupancy_ratio/_neural_impl.py",
    }


def _result_source_hashes() -> dict[str, str]:
    return {
        name: _file_sha256(path) for name, path in _result_source_paths().items()
    }


def _configuration_identity(args: argparse.Namespace, scope: str) -> dict[str, object]:
    """Return an immutable identity for resumable result-producing cells."""
    ignored = {"output_dir", "jobs", "resume"}
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in ignored
    }
    for key in ("fore_frozen_config", "fusion_g_cache", "fusion_pilot_manifest"):
        path_value = getattr(args, key, None)
        if path_value is None:
            continue
        path = Path(path_value)
        if path.is_file():
            payload[f"{key}_sha256"] = _file_sha256(path)
    identity_payload = json.loads(
        json.dumps(
            {
                "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
                "scope": scope,
                "arguments": payload,
                "result_source_sha256": _result_source_hashes(),
            },
            sort_keys=True,
            default=str,
        )
    )
    encoded = json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    configuration_sha256 = hashlib.sha256(encoded).hexdigest()
    return identity_payload | {
        "configuration_sha256": configuration_sha256,
        "fingerprint": configuration_sha256[:16],
    }


def _configuration_fingerprint(args: argparse.Namespace, scope: str) -> str:
    """Key resumable cells by configuration, frozen inputs, and live source."""
    return str(_configuration_identity(args, scope)["fingerprint"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_cell_identity(
    path: Path,
    *,
    configuration_identity: dict[str, object],
    cell: dict[str, object],
    checkpoint_exists: bool,
) -> None:
    """Create or verify the immutable identity adjacent to one checkpoint."""
    expected = configuration_identity | {"cell": cell}
    if path.exists():
        with path.open() as handle:
            observed = json.load(handle)
        if observed != expected:
            raise RuntimeError(
                f"Refusing checkpoint with mismatched immutable identity: {path}"
            )
        return
    if checkpoint_exists:
        raise RuntimeError(
            f"Refusing checkpoint without an immutable identity sidecar: {path}"
        )
    save_json(expected, path)


def _verify_expected_sha256(
    path: Path | None, expected: str | None, *, label: str
) -> None:
    if expected is None:
        return
    if path is None or not path.is_file():
        raise ValueError(f"{label} requires an existing locked input file.")
    observed = _file_sha256(path)
    if observed != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}."
        )


def _apply_locked_protocol_inputs(args: argparse.Namespace) -> None:
    """Verify pinned nuisance inputs and derive the fusion gate from its pilot."""
    _verify_expected_sha256(
        args.fore_frozen_config,
        args.expected_fore_sha256,
        label="Frozen FORE configuration",
    )
    _verify_expected_sha256(
        args.fusion_g_cache,
        args.expected_fusion_g_sha256,
        label="Frozen outcome regression",
    )
    if args.fusion_pilot_manifest is None:
        if args.expected_fusion_pilot_manifest_sha256 is not None:
            raise ValueError(
                "An expected fusion-pilot manifest hash requires "
                "--fusion-pilot-manifest."
            )
        return
    _verify_expected_sha256(
        args.fusion_pilot_manifest,
        args.expected_fusion_pilot_manifest_sha256,
        label="Data-fusion pilot manifest",
    )
    with args.fusion_pilot_manifest.open() as handle:
        pilot = json.load(handle)
    readiness = pilot.get("readiness")
    if pilot.get("mode") != "data-fusion-pilot" or not isinstance(readiness, dict):
        raise ValueError("The fusion-pilot manifest is not a locked pilot artifact.")
    if readiness.get("passed") is not True:
        raise ValueError("The locked data-fusion pilot did not pass its shift gate.")
    pilot_median_se = float(pilot["pilot_median_se"])
    recorded_median_se = float(readiness["pilot_median_se"])
    if not np.isfinite(pilot_median_se) or pilot_median_se <= 0.0:
        raise ValueError("The locked pilot median standard error is invalid.")
    if not np.isclose(pilot_median_se, recorded_median_se, rtol=0.0, atol=1e-15):
        raise ValueError("The fusion-pilot readiness record is internally inconsistent.")
    if args.fusion_pilot_median_se is not None and not np.isclose(
        float(args.fusion_pilot_median_se), pilot_median_se, rtol=0.0, atol=1e-15
    ):
        raise ValueError(
            "--fusion-pilot-median-se disagrees with the pinned pilot manifest."
        )
    args.fusion_pilot_median_se = pilot_median_se


def _apply_frozen_fore_config(args: argparse.Namespace) -> None:
    """Load the truth-blind pilot winner without changing stopping budgets."""
    if args.fore_frozen_config is None:
        return
    with args.fore_frozen_config.open() as handle:
        payload = json.load(handle)
    if payload.get("uses_oracle_truth") is not False:
        raise ValueError(
            "The frozen FORE configuration must explicitly record uses_oracle_truth=false."
        )
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("The frozen FORE configuration has no selected pilot winner.")
    hidden_dims = tuple(int(width) for width in selected["hidden_dims"])
    learning_rate = float(selected["learning_rate"])
    weight_decay = float(selected["weight_decay"])
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("The frozen FORE hidden dimensions are invalid.")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("The frozen FORE optimizer settings are invalid.")
    args.fore_hidden_sizes = list(hidden_dims)
    args.fore_learning_rate = learning_rate
    args.fore_weight_decay = weight_decay


def _environment_manifest() -> dict[str, object]:
    """Record the reproducibility-critical runtime versions."""
    packages = {}
    for name in ("numpy", "scipy", "torch", "scikit-learn", "lightgbm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("worktree_status_porcelain", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
        )
        git[key] = completed.stdout.rstrip() if completed.returncode == 0 else None
    source_dir = Path(__file__).resolve().parent
    source_paths = _result_source_paths() | {
        "assemble_paper_results.py": source_dir / "assemble_paper_results.py",
        "run_paper_confirmatory.sh": source_dir / "run_paper_confirmatory.sh",
    }
    source_hashes = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git": git,
        "local_source_sha256": source_hashes,
    }


def _validate_and_record_artifacts(
    output_dir: Path,
    required_paths: Sequence[Path],
    *,
    manifest_name: str,
) -> None:
    """Fail on missing/empty outputs and record hashes for reproducibility."""
    missing = [path for path in required_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "Missing or empty experiment artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    rows = []
    for path in required_paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    save_json(
        {"validated": True, "artifacts": rows},
        output_dir / manifest_name,
    )


def _load_single_run_cell(path: Path) -> list[SingleRunResult]:
    """Load one complete main-study cell from its machine-readable CSV."""
    if not path.exists():
        return []
    integer_fields = {"n", "seed"}
    string_fields = {
        "example_id",
        "ratio_mode",
        "nuisance_method",
        "fore_selected_iterations_by_fold",
        "fore_apbv_scores_by_fold",
        "signed_positive_iterations_by_fold",
        "signed_negative_iterations_by_fold",
        "signed_positive_apbv_scores_by_fold",
        "signed_negative_apbv_scores_by_fold",
        "failure_message",
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SingleRunResult(
            **{
                key: (
                    value
                    if key in string_fields
                    else int(value)
                    if key in integer_fields
                    else float(value)
                )
                for key, value in row.items()
            }
        )
        for row in rows
    ]


def _load_data_fusion_cell(path: Path) -> list[DataFusionRunResult]:
    """Load one complete data-fusion cell from its machine-readable CSV."""
    if not path.exists():
        return []
    integer_fields = {"n", "seed", "outcome_sample_size"}
    string_fields = {
        "fore_selected_iterations_by_fold",
        "fore_apbv_scores_by_fold",
        "failure_message",
        "data_fusion_policy_mode",
        "data_fusion_transition_mode",
        "data_fusion_g_mode",
        "data_fusion_ratio_mode",
        "normalization_policy_mode",
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        DataFusionRunResult(
            **{
                key: (
                    value
                    if key in string_fields
                    else int(value)
                    if key in integer_fields
                    else float(value)
                )
                for key, value in row.items()
            }
        )
        for row in rows
    ]


def _main_cell_is_complete(
    rows: Sequence[SingleRunResult],
    *,
    repetitions: int,
    example_id: str,
    n: int,
    ratio_mode: str,
    seed_offset: int = 0,
) -> bool:
    """Reject partial or mis-keyed main-study cell files."""
    return _main_cell_is_compatible(
        rows,
        repetitions=repetitions,
        example_id=example_id,
        n=n,
        ratio_mode=ratio_mode,
        seed_offset=seed_offset,
    ) and len(rows) == repetitions


def _main_cell_is_compatible(
    rows: Sequence[SingleRunResult],
    *,
    repetitions: int,
    example_id: str,
    n: int,
    ratio_mode: str,
    seed_offset: int = 0,
) -> bool:
    """Accept a unique deterministic subset that can be resumed safely."""
    expected_seeds = set(
        _main_replication_seeds(
            repetitions=repetitions,
            example_id=example_id,
            n=n,
            effective_seed_offset=seed_offset,
        )
    )
    return (
        len({row.seed for row in rows}) == len(rows)
        and {row.seed for row in rows}.issubset(expected_seeds)
        and all(
            row.example_id == example_id
            and row.n == n
            and row.ratio_mode == ratio_mode
            for row in rows
        )
    )


def _main_replication_seeds(
    *,
    repetitions: int,
    example_id: str,
    n: int,
    effective_seed_offset: int,
) -> list[int]:
    """Return deterministic main-study seeds, including the CLI base seed."""
    example_offsets = {"1a": 101, "1b": 202, "2": 303}
    return [
        10_000 * (rep + 1)
        + 97 * int(n)
        + example_offsets[example_id]
        + int(effective_seed_offset)
        for rep in range(int(repetitions))
    ]


def _data_fusion_cell_is_complete(
    rows: Sequence[DataFusionRunResult],
    *,
    repetitions: int,
    n: int,
    seed: int,
) -> bool:
    """Reject partial or mis-keyed data-fusion cell files."""
    return _data_fusion_cell_is_compatible(
        rows,
        repetitions=repetitions,
        n=n,
        seed=seed,
    ) and len(rows) == repetitions


def _data_fusion_cell_is_compatible(
    rows: Sequence[DataFusionRunResult],
    *,
    repetitions: int,
    n: int,
    seed: int,
) -> bool:
    """Accept a unique deterministic data-fusion subset for resumption."""
    expected_seeds = {
        seed + 100_003 * repetition + 97 * n
        for repetition in range(repetitions)
    }
    return (
        len({row.seed for row in rows}) == len(rows)
        and {row.seed for row in rows}.issubset(expected_seeds)
        and all(row.n == n for row in rows)
    )


def _atomic_save_main_cell(rows: Sequence[SingleRunResult], path: Path) -> None:
    """Atomically replace a main-study checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_results_csv(rows, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_data_fusion_cell(
    rows: Sequence[DataFusionRunResult], path: Path
) -> None:
    """Atomically replace a data-fusion checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_summary_csv([result.as_dict() for result in rows], temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _budgets_are_predeclared(encoded: str, budgets: Sequence[int]) -> bool:
    """Validate fold-level stopping choices, including failure-row empties."""
    if not encoded:
        return False
    try:
        values = json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        return False
    allowed = {int(value) for value in budgets}
    return bool(values) and all(
        isinstance(value, (int, float)) and int(value) in allowed
        for value in values
    )


def run_mc(oracle: JRSSBOracle, output_dir: Path, args: argparse.Namespace) -> None:
    unknown_examples = set(args.examples) - {"1a", "1b", "2"}
    if unknown_examples:
        raise ValueError(f"Unknown examples: {sorted(unknown_examples)}")
    if args.ratio_mode == "neural-fore" and "2" in args.examples:
        raise ValueError(
            "The paper-facing neural-FORE study uses Examples 1a/1b. "
            "Use a data-fusion mode for the replacement practical study; "
            "legacy Example 2 remains available with --ratio-mode oracle or coarse-estimated."
        )
    if args.sample_sizes is not None:
        sample_sizes = tuple(args.sample_sizes)
    elif args.mode == "fore-smoke":
        sample_sizes = (2500,)
    else:
        sample_sizes = tuple(oracle.config.mc_sample_sizes)
    if args.repetitions is not None:
        repetitions = args.repetitions
    elif args.mode == "fore-smoke":
        repetitions = 2
    else:
        repetitions = oracle.config.mc_repetitions
    configuration_identity = _configuration_identity(args, "main-inference")
    fingerprint = str(configuration_identity["fingerprint"])
    cell_dir = output_dir / "cells" / fingerprint
    results: list[SingleRunResult] = []
    effective_seed_offset = int(args.seed) + int(args.replication_seed_offset)
    for example_id in args.examples:
        for n in sample_sizes:
            cell_path = cell_dir / f"example_{example_id}_n{n}.csv"
            identity_path = cell_path.with_suffix(".identity.json")
            _ensure_cell_identity(
                identity_path,
                configuration_identity=configuration_identity,
                cell={"example_id": example_id, "n": int(n)},
                checkpoint_exists=cell_path.exists(),
            )
            cell = _load_single_run_cell(cell_path) if args.resume else []
            if not _main_cell_is_compatible(
                cell,
                repetitions=repetitions,
                example_id=example_id,
                n=n,
                ratio_mode=args.ratio_mode,
                seed_offset=effective_seed_offset,
            ):
                cell = []
            seed_order = _main_replication_seeds(
                repetitions=repetitions,
                example_id=example_id,
                n=n,
                effective_seed_offset=effective_seed_offset,
            )
            rows_by_seed = {row.seed: row for row in cell}
            missing_indices = [
                index
                for index, replication_seed in enumerate(seed_order)
                if replication_seed not in rows_by_seed
            ]

            def checkpoint(result: SingleRunResult) -> None:
                rows_by_seed[result.seed] = result
                ordered = [
                    rows_by_seed[replication_seed]
                    for replication_seed in seed_order
                    if replication_seed in rows_by_seed
                ]
                _atomic_save_main_cell(ordered, cell_path)
                if len(ordered) == 1 or len(ordered) % 10 == 0 or len(ordered) == repetitions:
                    print(
                        f"[main] {example_id} n={n}: "
                        f"{len(ordered)}/{repetitions} replications checkpointed",
                        flush=True,
                    )

            if missing_indices:
                run_monte_carlo(
                    oracle=oracle,
                    sample_sizes=(n,),
                    repetitions=repetitions,
                    example_ids=(example_id,),
                    ratio_mode=args.ratio_mode,
                    jobs=args.jobs,
                    seed_offset=effective_seed_offset,
                    replication_indices=missing_indices,
                    on_result=checkpoint,
                )
            cell = [rows_by_seed[replication_seed] for replication_seed in seed_order]
            results.extend(cell)
    summary_rows = summarize_results(results)
    save_results_csv(results, output_dir / "results.csv")
    save_summary_csv(summary_rows, output_dir / "summary.csv")
    write_main_inference_latex_table(
        summary_rows, output_dir / "main_inference_table.tex"
    )
    maybe_plot(summary_rows, output_dir)
    smoke_acceptance = None
    if args.mode == "fore-smoke":
        diagnostic_oracle = JRSSBOracle(
            replace(
                oracle.config,
                main_grid_points=7,
                coarse_grid_points=5,
                max_iterations=500,
                use_oracle_cache=False,
            )
        )
        signed_oracle_diagnostic = (
            diagnostic_oracle.signed_jordan_oracle_diagnostic()
            | {"oracle_grid_points_per_dimension": 7}
        )
        save_json(
            signed_oracle_diagnostic,
            output_dir / "oracle_signed_ratio_diagnostic.json",
        )
        smoke_acceptance = {
            "finite_estimates": bool(
                all(
                    np.all(
                        np.isfinite(
                            [
                                row.plugin_estimate,
                                row.if_estimate,
                                row.estimated_se,
                                row.fore_apbv_score,
                            ]
                        )
                    )
                    for row in results
                )
            ),
            "no_ratio_failures": bool(
                all(row.ratio_failure == 0.0 for row in results)
            ),
            "normalized_mass_within_5_percent": bool(
                all(abs(row.fore_normalized_mass - 1.0) <= 0.05 for row in results)
            ),
            "positive_effective_sample_size": bool(
                all(row.weight_ess > 0.0 for row in results)
            ),
            "selected_budget_is_predeclared": bool(
                all(
                    _budgets_are_predeclared(
                        row.fore_selected_iterations_by_fold,
                        args.fore_iteration_budgets,
                    )
                    for row in results
                )
            ),
            "signed_jordan_matches_oracle_grid": bool(
                signed_oracle_diagnostic["signed_jordan_ratio_sup_error"] < 1e-8
            ),
        }
        smoke_acceptance["passed"] = bool(all(smoke_acceptance.values()))
        save_json(smoke_acceptance, output_dir / "smoke_acceptance.json")
    required = [
        output_dir / "run_config.json",
        output_dir / "environment.json",
        output_dir / "results.csv",
        output_dir / "summary.csv",
        output_dir / "main_inference_table.tex",
        *sorted(cell_dir.glob("*.csv")),
        *sorted(cell_dir.glob("*.identity.json")),
    ]
    if args.ratio_mode == "neural-fore":
        required.append(output_dir / "selection_manifest.json")
    if args.fore_frozen_config is not None:
        required.append(args.fore_frozen_config)
    if args.mode == "fore-smoke":
        required.extend(
            [
                output_dir / "smoke_acceptance.json",
                output_dir / "oracle_signed_ratio_diagnostic.json",
            ]
        )
    _validate_and_record_artifacts(
        output_dir,
        required,
        manifest_name="artifact_manifest.json",
    )
    if smoke_acceptance is not None and not smoke_acceptance["passed"]:
        raise RuntimeError("The neural-FORE smoke acceptance checks failed.")


_FORE_PILOT_WORKER_ORACLE: JRSSBOracle | None = None
_FORE_PILOT_WORKER_CANDIDATES: tuple[FORECandidateConfig, ...] = ()
_FORE_PILOT_WORKER_OPTIONS: FOREFitOptions | None = None


def _serialize_apbv_selection(
    selection: APBVSelectionResult | None,
) -> dict[str, object] | None:
    if selection is None:
        return None
    return {
        "candidate_ids": list(selection.candidate_ids),
        "worst_case_scores": selection.worst_case_scores.tolist(),
        "selected_index": int(selection.selected_index),
    }


def _deserialize_apbv_selection(payload: dict[str, object]) -> APBVSelectionResult:
    candidate_ids = tuple(str(value) for value in payload["candidate_ids"])
    scores = np.asarray(payload["worst_case_scores"], dtype=float)
    selected_index = int(payload["selected_index"])
    if scores.shape != (len(candidate_ids),) or not np.all(np.isfinite(scores)):
        raise ValueError("A cached pilot selection has invalid A-PBV scores.")
    if not 0 <= selected_index < len(candidate_ids):
        raise ValueError("A cached pilot selection has an invalid selected index.")
    return APBVSelectionResult(
        candidate_ids=candidate_ids,
        score_matrix=np.zeros((len(candidate_ids), len(candidate_ids)), dtype=float),
        worst_case_scores=scores,
        selected_index=selected_index,
    )


def _fore_pilot_event(
    *,
    selection: APBVSelectionResult | None,
    component: str,
    example_id: str,
    n: int,
    repetition: int,
    seed: int,
    fold: int,
    outer_train_n: int,
    component_mass: float | None = None,
) -> dict[str, object]:
    return {
        "component": component,
        "example_id": example_id,
        "n": int(n),
        "repetition": int(repetition),
        "seed": int(seed),
        "fold": int(fold),
        "outer_train_n": int(outer_train_n),
        "component_mass": component_mass,
        "skipped": selection is None,
        "selection": _serialize_apbv_selection(selection),
    }


def _evaluate_fore_pilot_task(
    *,
    oracle: JRSSBOracle,
    candidates: tuple[FORECandidateConfig, ...],
    options: FOREFitOptions,
    task: tuple[str, int, int, int],
) -> list[dict[str, object]]:
    """Evaluate both outer folds for one truth-blind pilot dataset."""
    example_id, n, repetition, event_seed = task
    data = oracle.sample_stationary_transitions(n=n, seed=event_seed)
    all_indices = np.arange(n)
    records: list[dict[str, object]] = []
    for fold_number, outer_eval_idx in enumerate(
        fold_splits(n, seed=event_seed + 503, n_folds=2)
    ):
        outer_train_mask = np.ones(n, dtype=bool)
        outer_train_mask[outer_eval_idx] = False
        outer_train_idx = all_indices[outer_train_mask]
        states = data["states"][outer_train_idx]
        actions = data["actions"][outer_train_idx]
        next_states = data["next_states"][outer_train_idx]
        fold_seed = event_seed + 10_007 * (fold_number + 1)
        split = deterministic_three_way_split(states.shape[0], fold_seed + 701)
        q_grid = None
        v_grid = None
        if example_id == "1a":
            target_policy = make_policy_adapter(
                oracle, deterministic_fn=oracle._fixed_policy_probs
            )
        else:
            fit_idx = split[0]
            policy_hat = fit_behavior_policy_example1b(
                oracle=oracle,
                states=states[fit_idx],
                actions=actions[fit_idx],
                seed=fold_seed + 17,
            )
            reward_grid = np.log(
                np.clip(
                    policy_hat.predict_proba(oracle.main_grid.states),
                    1e-8,
                    None,
                )
            )
            _, _, pi_star_grid = fit_soft_value_from_reward(
                oracle,
                reward_grid=reward_grid,
                tau=oracle.config.tau_star,
                allowed_mask=oracle.allowed_mask,
            )
            q_grid, v_grid = oracle.evaluate_policy(
                reward_grid,
                pi_star_grid,
                gamma=oracle.config.gamma_behavior,
            )
            target_policy = make_policy_adapter(oracle, policy_grid=pi_star_grid)
        selected = fit_selected_fore_ratio(
            states=states,
            actions=actions,
            next_states=next_states,
            target_policy=target_policy,
            gamma=oracle.config.gamma_behavior,
            n_actions=oracle.config.n_actions,
            candidates=candidates,
            seed=fold_seed + 809,
            options=options,
            split=split,
        )
        records.append(
            _fore_pilot_event(
                selection=selected.selection,
                component="ordinary",
                example_id=example_id,
                n=n,
                repetition=repetition,
                seed=event_seed,
                fold=fold_number,
                outer_train_n=states.shape[0],
            )
        )
        if example_id != "1b":
            continue
        if q_grid is None or v_grid is None:
            raise RuntimeError("Example 1b pilot values were not constructed.")
        q_all = oracle.action_values(states, q_grid)
        v_all = oracle.state_values(states, v_grid)
        d_all = selected.selection_ratio.predict_unnormalized(states, actions)
        d_all_refit = selected.ratio.predict_unnormalized(states, actions)
        signed = fit_selected_signed_fore_ratio(
            states=states,
            actions=actions,
            next_states=next_states,
            source_weights=d_all
            * (q_all[np.arange(states.shape[0]), actions] - v_all),
            refit_source_weights=d_all_refit
            * (q_all[np.arange(states.shape[0]), actions] - v_all),
            target_policy=target_policy,
            gamma=oracle.config.gamma_behavior,
            n_actions=oracle.config.n_actions,
            candidates=candidates,
            seed=fold_seed + 1_809,
            split=split,
            options=options,
            mass_tolerance=oracle.config.fore_signed_mass_tolerance,
        )
        for component, selection, mass in (
            (
                "signed-positive",
                signed.positive_selection,
                signed.positive_mass,
            ),
            (
                "signed-negative",
                signed.negative_selection,
                signed.negative_mass,
            ),
        ):
            records.append(
                _fore_pilot_event(
                    selection=selection,
                    component=component,
                    example_id=example_id,
                    n=n,
                    repetition=repetition,
                    seed=event_seed,
                    fold=fold_number,
                    outer_train_n=states.shape[0],
                    component_mass=float(mass),
                )
            )
    return records


def _initialize_fore_pilot_worker(
    config: JRSSBConfig,
    candidates: tuple[FORECandidateConfig, ...],
    options: FOREFitOptions,
) -> None:
    global _FORE_PILOT_WORKER_ORACLE
    global _FORE_PILOT_WORKER_CANDIDATES
    global _FORE_PILOT_WORKER_OPTIONS
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    _FORE_PILOT_WORKER_ORACLE = JRSSBOracle(config)
    _FORE_PILOT_WORKER_CANDIDATES = candidates
    _FORE_PILOT_WORKER_OPTIONS = options


def _run_fore_pilot_worker(
    task: tuple[str, int, int, int],
) -> list[dict[str, object]]:
    if _FORE_PILOT_WORKER_ORACLE is None or _FORE_PILOT_WORKER_OPTIONS is None:
        raise RuntimeError("The FORE pilot worker was not initialized.")
    return _evaluate_fore_pilot_task(
        oracle=_FORE_PILOT_WORKER_ORACLE,
        candidates=_FORE_PILOT_WORKER_CANDIDATES,
        options=_FORE_PILOT_WORKER_OPTIONS,
        task=task,
    )


def _fore_pilot_cell_is_complete(
    payload: object,
    *,
    task: tuple[str, int, int, int],
    candidate_ids: tuple[str, ...],
) -> bool:
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        return False
    if tuple(payload.get("task", ())) != task:
        return False
    records = payload.get("events")
    expected_count = 2 if task[0] == "1a" else 6
    if not isinstance(records, list) or len(records) != expected_count:
        return False
    expected_components = (
        ["ordinary", "ordinary"]
        if task[0] == "1a"
        else ["ordinary", "signed-positive", "signed-negative"] * 2
    )
    if [record.get("component") for record in records] != expected_components:
        return False
    for record in records:
        selection = record.get("selection")
        if selection is None:
            if record.get("component") == "ordinary":
                return False
            continue
        if tuple(selection.get("candidate_ids", ())) != candidate_ids:
            return False
    return True


def run_fore_selection_pilot_mode(output_dir: Path, args: argparse.Namespace) -> None:
    """Run the predeclared truth-blind A-PBV hyperparameter pilot."""
    config = build_config(args)
    oracle = JRSSBOracle(config)
    candidates = paper_pilot_candidates()
    options = FOREFitOptions(
        batch_size=args.fore_batch_size,
        optimizer_steps=args.fore_optimizer_steps,
        target_action_draws=args.fore_target_action_draws,
        logit_clip=args.fore_logit_clip,
        grad_clip_norm=args.fore_grad_clip_norm,
        device=args.fore_device,
    )
    sample_sizes = tuple(args.sample_sizes) if args.sample_sizes is not None else (2500, 10000)
    repetitions = args.fore_pilot_repetitions if args.repetitions is None else args.repetitions
    examples = tuple(example for example in args.examples if example in {"1a", "1b"})
    if not examples:
        raise ValueError("fore-selection-pilot requires Example 1a and/or 1b.")
    tasks = [
        (
            example_id,
            int(n),
            repetition,
            args.seed
            + 100_003 * repetition
            + 97 * int(n)
            + (101 if example_id == "1a" else 202),
        )
        for example_id in examples
        for n in sample_sizes
        for repetition in range(repetitions)
    ]
    fingerprint = _configuration_fingerprint(args, "fore-selection-pilot")
    cell_dir = output_dir / "cells" / fingerprint
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    task_records: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    missing_tasks = []
    for task in tasks:
        cell_path = cell_dir / f"{task[0]}_n{task[1]}_rep{task[2]:03d}.json"
        payload = None
        if args.resume and cell_path.exists():
            with cell_path.open() as handle:
                payload = json.load(handle)
        if _fore_pilot_cell_is_complete(
            payload,
            task=task,
            candidate_ids=candidate_ids,
        ):
            task_records[task] = payload["events"]
        else:
            missing_tasks.append(task)

    if args.jobs <= 1:
        evaluated = (
            _evaluate_fore_pilot_task(
                oracle=oracle,
                candidates=candidates,
                options=options,
                task=task,
            )
            for task in missing_tasks
        )
        iterator = zip(missing_tasks, evaluated)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_initialize_fore_pilot_worker,
            initargs=(config, candidates, options),
        )
        iterator = zip(missing_tasks, executor.map(_run_fore_pilot_worker, missing_tasks))
    try:
        for task, records in iterator:
            task_records[task] = records
            save_json(
                {"complete": True, "task": task, "events": records},
                cell_dir / f"{task[0]}_n{task[1]}_rep{task[2]:03d}.json",
            )
            completed = len(task_records)
            if completed == 1 or completed % 5 == 0 or completed == len(tasks):
                print(
                    f"[FORE pilot] {completed}/{len(tasks)} cells checkpointed",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    serialized_events = [record for task in tasks for record in task_records[task]]
    events = []
    selection_rows = []
    for event_index, record in enumerate(serialized_events):
        selection_payload = record["selection"]
        if selection_payload is None:
            continue
        selection = _deserialize_apbv_selection(selection_payload)
        events.append((candidates, selection))
        for candidate, row in zip(
            candidates,
            selection.rows(str(record["component"])),
        ):
            selection_rows.append(
                row
                | {
                    "event_index": event_index,
                    "example_id": record["example_id"],
                    "n": record["n"],
                    "repetition": record["repetition"],
                    "seed": record["seed"],
                    "fold": record["fold"],
                    "outer_train_n": record["outer_train_n"],
                    "component_mass": record["component_mass"],
                    "hidden_dims": "x".join(map(str, candidate.hidden_dims)),
                    "learning_rate": candidate.learning_rate,
                    "weight_decay": candidate.weight_decay,
                    "iterations": candidate.num_iterations,
                }
            )
    decision = aggregate_pilot_selections(events)
    frozen_config = {
        key: value for key, value in decision.items() if key != "event_rows"
    }
    save_summary_csv(selection_rows, output_dir / "selection_rows.csv")
    save_summary_csv(decision["summary_rows"], output_dir / "selection_summary.csv")
    save_json(frozen_config, output_dir / "frozen_fore_config.json")
    save_json({"events": serialized_events}, output_dir / "selection_events.json")
    plot_apbv_selection_behavior(
        decision,
        serialized_events,
        output_dir / "apbv_selection_behavior.png",
    )
    manifest = selection_manifest(candidates, options)
    manifest["protocol"] = {
        "examples": examples,
        "sample_sizes": sample_sizes,
        "repetitions": repetitions,
        "outer_folds": 2,
        "fit_validation_signed_split": [0.60, 0.20, 0.20],
        "validation_empirical_terms": "same-held-out-rows",
        "exact_validation_action_summation": True,
        "resumable_cell_directory": str(cell_dir.resolve()),
    }
    save_json(manifest, output_dir / "pilot_manifest.json")
    required = [
        output_dir / "run_config.json",
        output_dir / "environment.json",
        output_dir / "selection_rows.csv",
        output_dir / "selection_summary.csv",
        output_dir / "selection_events.json",
        output_dir / "frozen_fore_config.json",
        output_dir / "pilot_manifest.json",
        *sorted(cell_dir.glob("*.json")),
    ]
    if (output_dir / "apbv_selection_behavior.png").exists():
        required.append(output_dir / "apbv_selection_behavior.png")
    _validate_and_record_artifacts(
        output_dir,
        required,
        manifest_name="artifact_manifest.json",
    )


def _load_or_fit_outcome_regression(
    oracle: JRSSBOracle,
    output_dir: Path,
    args: argparse.Namespace,
) -> FrozenOutcomeRegression:
    cache_path = args.fusion_g_cache or (output_dir / "frozen_outcome_regression.pkl")
    if cache_path.exists():
        result = FrozenOutcomeRegression.load(cache_path)
        expected = (
            args.outcome_sample_size,
            args.outcome_regression_seed,
            args.outcome_noise_sd,
        )
        observed = (result.sample_size, result.seed, result.outcome_noise_sd)
        if observed[:2] != expected[:2] or not np.isclose(observed[2], expected[2]):
            raise ValueError(
                "The cached outcome regression metadata does not match the requested "
                f"(sample size, seed, noise SD): {observed} versus {expected}."
            )
        if result.normalization_policy_mode != "known-uniform":
            raise ValueError(
                "The cached outcome regression uses normalization policy "
                f"{result.normalization_policy_mode!r}; expected 'known-uniform'. "
                "Fit a new auxiliary regression for the randomized outcome source."
            )
        if not np.isclose(result.target_gamma, oracle.config.data_fusion_target_gamma):
            raise ValueError(
                "The cached outcome regression's target-gamma diagnostic is stale: "
                f"{result.target_gamma} versus {oracle.config.data_fusion_target_gamma}."
            )
        expected_design = (
            "uniform-domain",
            "outcome-summary-lbfgs-tanh-ensemble-v4",
        )
        observed_design = (result.source_state_mode, result.outcome_model_family)
        if observed_design != expected_design:
            raise ValueError(
                "The cached outcome regression design is stale: "
                f"{observed_design!r}; expected {expected_design!r}."
            )
        return result
    result = fit_frozen_outcome_regression(
        oracle,
        sample_size=args.outcome_sample_size,
        seed=args.outcome_regression_seed,
        outcome_noise_sd=args.outcome_noise_sd,
    )
    result.save(cache_path)
    return result


def run_data_fusion_mode(output_dir: Path, args: argparse.Namespace) -> None:
    """Run a staged data-fusion study with one frozen outcome regression."""
    oracle = JRSSBOracle(build_config(args))
    outcome_regression = _load_or_fit_outcome_regression(oracle, output_dir, args)
    if args.sample_sizes is not None:
        sample_sizes = tuple(args.sample_sizes)
    elif args.mode == "data-fusion-smoke":
        sample_sizes = (2500,)
    elif args.mode == "data-fusion-pilot":
        sample_sizes = (2500, 10000)
    else:
        sample_sizes = (2500, 5000, 10000)
    if args.repetitions is not None:
        repetitions = args.repetitions
    elif args.mode == "data-fusion-smoke":
        repetitions = 2
    elif args.mode == "data-fusion-pilot":
        repetitions = args.fusion_pilot_repetitions
    else:
        repetitions = 300

    readiness = None
    if args.mode == "data-fusion-confirmatory":
        if args.fusion_pilot_median_se is None:
            raise ValueError(
                "data-fusion-confirmatory requires --fusion-pilot-median-se from the locked pilot."
            )
        readiness = data_fusion_readiness(
            outcome_regression, args.fusion_pilot_median_se
        )
        if not readiness["passed"]:
            save_json(readiness, output_dir / "data_fusion_readiness.json")
            raise RuntimeError(
                "The frozen g regression failed the predeclared estimand-shift gate."
            )

    configuration_identity = _configuration_identity(args, "data-fusion")
    fingerprint = str(configuration_identity["fingerprint"])
    cell_dir = output_dir / "cells" / fingerprint
    results: list[DataFusionRunResult] = []
    for n in sample_sizes:
        cell_path = cell_dir / f"data_fusion_n{n}.csv"
        identity_path = cell_path.with_suffix(".identity.json")
        _ensure_cell_identity(
            identity_path,
            configuration_identity=configuration_identity,
            cell={"n": int(n)},
            checkpoint_exists=cell_path.exists(),
        )
        cell = _load_data_fusion_cell(cell_path) if args.resume else []
        if not _data_fusion_cell_is_compatible(
            cell,
            repetitions=repetitions,
            n=n,
            seed=args.seed,
        ):
            cell = []
        seed_order = [
            args.seed + 100_003 * repetition + 97 * n
            for repetition in range(repetitions)
        ]
        rows_by_seed = {result.seed: result for result in cell}
        missing_indices = [
            index
            for index, replication_seed in enumerate(seed_order)
            if replication_seed not in rows_by_seed
        ]

        def checkpoint(result: DataFusionRunResult) -> None:
            rows_by_seed[result.seed] = result
            ordered = [
                rows_by_seed[replication_seed]
                for replication_seed in seed_order
                if replication_seed in rows_by_seed
            ]
            _atomic_save_data_fusion_cell(ordered, cell_path)
            if len(ordered) == 1 or len(ordered) % 10 == 0 or len(ordered) == repetitions:
                print(
                    f"[data fusion] n={n}: "
                    f"{len(ordered)}/{repetitions} replications checkpointed",
                    flush=True,
                )

        if missing_indices:
            run_data_fusion_monte_carlo(
                oracle=oracle,
                outcome_regression=outcome_regression,
                sample_sizes=(n,),
                repetitions=repetitions,
                seed=args.seed,
                jobs=args.jobs,
                replication_indices=missing_indices,
                on_result=checkpoint,
            )
        cell = [rows_by_seed[replication_seed] for replication_seed in seed_order]
        results.extend(cell)
    summary = summarize_data_fusion_results(results)
    save_summary_csv([result.as_dict() for result in results], output_dir / "data_fusion_results.csv")
    save_summary_csv(summary, output_dir / "data_fusion_summary.csv")
    write_data_fusion_latex_table(
        summary, output_dir / "data_fusion_table.tex"
    )
    maybe_plot_data_fusion(summary, output_dir)
    finite_standard_errors = np.asarray(
        [result.estimated_se for result in results if np.isfinite(result.estimated_se)],
        dtype=float,
    )
    if finite_standard_errors.size == 0:
        raise RuntimeError("The data-fusion stage produced no finite standard errors.")
    pilot_median_se = float(np.median(finite_standard_errors))
    if args.mode == "data-fusion-pilot":
        readiness = data_fusion_readiness(outcome_regression, pilot_median_se)
    smoke_acceptance = None
    if args.mode == "data-fusion-smoke":
        neural_ratio = args.data_fusion_ratio_mode == "neural-fore"
        smoke_acceptance = {
            "finite_estimates": bool(
                all(
                    np.all(
                        np.isfinite(
                            [
                                result.plugin_estimate,
                                result.if_estimate,
                                result.estimated_se,
                            ]
                            + ([result.fore_apbv_score] if neural_ratio else [])
                        )
                    )
                    for result in results
                )
            ),
            "no_ratio_failures": bool(
                all(result.ratio_failure == 0.0 for result in results)
            ),
            "normalized_mass_within_5_percent": bool(
                all(
                    abs(result.fore_normalized_mass - 1.0) <= 0.05
                    for result in results
                )
            ),
            "positive_effective_sample_size": bool(
                all(result.ratio_ess > 0.0 for result in results)
            ),
            "selected_budget_is_predeclared": bool(
                not neural_ratio
                or all(
                    _budgets_are_predeclared(
                        result.fore_selected_iterations_by_fold,
                        args.fore_iteration_budgets,
                    )
                    for result in results
                )
            ),
        }
        smoke_acceptance["passed"] = bool(all(smoke_acceptance.values()))
        save_json(smoke_acceptance, output_dir / "smoke_acceptance.json")
    save_json(
        {
            "mode": args.mode,
            "conditional_on_frozen_g": True,
            "joint_g_uncertainty_covered": False,
            "outcome_sample_size": outcome_regression.sample_size,
            "outcome_regression_seed": outcome_regression.seed,
            "normalization_policy": {
                "mode": outcome_regression.normalization_policy_mode,
                "known_action_probabilities": [0.25, 0.25, 0.25, 0.25],
                "action_recorded_in_analysis_source": False,
            },
            "outcome_source_state_mode": outcome_regression.source_state_mode,
            "outcome_model_family": outcome_regression.outcome_model_family,
            "outcome_regression_selection": {
                "criterion": "one-SE held-out outcome mean-squared error",
                "selected_config": outcome_regression.selected_outcome_config,
                "selected_validation_mse": outcome_regression.outcome_validation_mse,
                "candidate_scores": outcome_regression.outcome_candidate_scores,
                "uses_oracle_truth": False,
            },
            "behavior_policy_selection": {
                "method": (
                    "observed logging probabilities"
                    if args.data_fusion_policy_mode == "known-logging"
                    else "one-SE polynomial-sieve multinomial logit"
                ),
                "criterion": "held-out observed-action negative log likelihood",
                "candidate_degrees": [2, 3, 4],
                "candidate_c": [0.01, 0.1, 1.0, 10.0],
                "mode": args.data_fusion_policy_mode,
                "probability_floor": args.data_fusion_probability_floor,
                "uses_transition_truth": False,
                "uses_reward_truth": False,
                "uses_estimand_truth": False,
                "logging_probabilities_observed": (
                    args.data_fusion_policy_mode == "known-logging"
                ),
            },
            "transition_selection": {
                "method": "action-specific degree-3 Gaussian transition sieve",
                "criterion": "held-out next-state mean-squared error",
                "candidate_ridge_alpha": [0.01, 0.1, 1.0, 10.0],
                "uses_transition_truth": False,
                "uses_reward_truth": False,
                "uses_estimand_truth": False,
                "mode": args.data_fusion_transition_mode,
            },
            "g_mode": args.data_fusion_g_mode,
            "ratio_mode": args.data_fusion_ratio_mode,
            "repeated_crossfit_splits": args.data_fusion_repeated_splits,
            "g_rmse_oracle_diagnostic": outcome_regression.g_rmse,
            "g_estimand_shift_oracle_diagnostic": outcome_regression.estimand_shift,
            "pilot_median_se": pilot_median_se,
            "readiness": readiness,
        },
        output_dir / "data_fusion_manifest.json",
    )
    required = [
        output_dir / "run_config.json",
        output_dir / "environment.json",
        output_dir / "data_fusion_results.csv",
        output_dir / "data_fusion_summary.csv",
        output_dir / "data_fusion_table.tex",
        output_dir / "data_fusion_manifest.json",
        *sorted(cell_dir.glob("*.csv")),
        *sorted(cell_dir.glob("*.identity.json")),
    ]
    if args.data_fusion_ratio_mode == "neural-fore":
        required.append(output_dir / "selection_manifest.json")
    if (output_dir / "data_fusion_performance.png").exists():
        required.append(output_dir / "data_fusion_performance.png")
    required.append(
        args.fusion_g_cache
        or (output_dir / "frozen_outcome_regression.pkl")
    )
    if args.fusion_pilot_manifest is not None:
        required.append(args.fusion_pilot_manifest)
    if args.mode == "data-fusion-smoke":
        required.append(output_dir / "smoke_acceptance.json")
    _validate_and_record_artifacts(
        output_dir,
        required,
        manifest_name="artifact_manifest.json",
    )
    if smoke_acceptance is not None and not smoke_acceptance["passed"]:
        raise RuntimeError("The data-fusion smoke acceptance checks failed.")


def run_example2_method_pilot(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    sample_sizes = args.sample_sizes if args.sample_sizes is not None else [2500, 5000, 10000]
    repetitions = 100 if args.repetitions is None else args.repetitions
    study = run_example2_method_selection(
        base_config=config,
        methods=args.example2_methods,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["selection"], output_dir / "selection.json")
    maybe_plot(study["summary"], output_dir)


def run_example2_paper_comparison_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    sample_sizes = args.sample_sizes if args.sample_sizes is not None else [5000, 10000]
    repetitions = args.shakedown_reps if args.repetitions is None else args.repetitions
    study = run_example2_paper_comparison(
        base_config=config,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_summary_csv(study["comparison"], output_dir / "comparison_table.csv")
    save_json(study["selection"], output_dir / "selection.json")
    save_json({"paper_methods": study["paper_methods"]}, output_dir / "paper_methods.json")
    maybe_plot(study["summary"], output_dir)


def run_example2_semi_oracle(output_dir: Path, args: argparse.Namespace) -> None:
    oracle = JRSSBOracle(build_config(args))
    rows = run_example2_semi_oracle_audit(
        oracle=oracle,
        n=args.example2_semi_oracle_n,
        seed=args.seed,
        ratio_mode=args.ratio_mode,
    )
    save_json({"rows": rows}, output_dir / "semi_oracle.json")


def run_example1b_policy_audit_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_example1b_policy_audit(
        base_config=config,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps if args.repetitions is None else args.repetitions,
    )
    save_summary_csv(study["rows"], output_dir / "rows.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["soft_policy_self_check"], output_dir / "soft_policy_self_check.json")


def run_example1a_policy_audit_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_example1a_policy_audit(
        base_config=config,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps if args.repetitions is None else args.repetitions,
    )
    save_summary_csv(study["rows"], output_dir / "rows.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")


def run_example2_nuisance_audit_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_example2_nuisance_audit(
        base_config=config,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps if args.repetitions is None else args.repetitions,
    )
    save_summary_csv(study["rows"], output_dir / "rows.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")


def run_example2_smoke(output_dir: Path, args: argparse.Namespace) -> None:
    oracle = JRSSBOracle(build_config(args))
    seeds = tuple(args.seed + offset for offset in range(args.repetitions)) if args.repetitions is not None else (201, 202, 203)
    study = run_example2_smoke_check(
        oracle=oracle,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        seeds=seeds,
        ratio_mode=args.ratio_mode,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["acceptance"], output_dir / "acceptance.json")
    maybe_plot(study["summary"], output_dir)


def run_example2_shakedown_mode(output_dir: Path, args: argparse.Namespace) -> None:
    oracle = JRSSBOracle(build_config(args))
    study = run_example2_shakedown(
        oracle=oracle,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["checks"], output_dir / "checks.json")
    maybe_plot(study["summary"], output_dir)


def main() -> None:
    args = parse_args()
    oracle_diagnostic_requested = (
        args.ratio_mode == "oracle-adaptive"
        or args.example1a_policy_estimator == "oracle"
        or args.example1b_policy_estimator == "oracle"
        or args.data_fusion_policy_mode == "oracle"
        or args.data_fusion_transition_mode == "oracle"
        or args.data_fusion_g_mode == "oracle"
        or args.data_fusion_ratio_mode == "oracle-adaptive"
    )
    if oracle_diagnostic_requested and not args.allow_oracle_diagnostics:
        raise ValueError(
            "Oracle component diagnostics require --allow-oracle-diagnostics; "
            "they are labeled audits and cannot be promoted by truth-based selection."
        )
    locked_fore_modes = {
        "monte-carlo",
        "data-fusion-pilot",
        "data-fusion-confirmatory",
    }
    if args.mode in locked_fore_modes and args.fore_frozen_config is None:
        needs_frozen_fore = (
            args.ratio_mode == "neural-fore"
            if not args.mode.startswith("data-fusion")
            else args.data_fusion_ratio_mode == "neural-fore"
        )
        if needs_frozen_fore:
            raise ValueError(
                f"{args.mode} requires --fore-frozen-config from the truth-blind "
                "fore-selection-pilot."
            )
    _apply_frozen_fore_config(args)
    _apply_locked_protocol_inputs(args)
    if args.paper_protocol_id is not None and args.paper_protocol_id != PAPER_PROTOCOL_ID:
        raise ValueError(
            f"Unknown paper protocol {args.paper_protocol_id!r}; "
            f"expected {PAPER_PROTOCOL_ID!r}."
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment_manifest()
    save_json(
        {
            "config": vars(args),
            "resolved_behavior_policy_design": resolved_behavior_policy_design(args),
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "result_source_sha256": _result_source_hashes(),
        },
        output_dir / "run_config.json",
    )
    save_json(environment, output_dir / "environment.json")
    uses_neural_fore = (
        args.data_fusion_ratio_mode == "neural-fore"
        if args.mode.startswith("data-fusion")
        else args.ratio_mode == "neural-fore"
    )
    if uses_neural_fore:
        candidates = paper_early_stopping_candidates(
            hidden_dims=tuple(args.fore_hidden_sizes),
            learning_rate=args.fore_learning_rate,
            weight_decay=args.fore_weight_decay,
            iteration_budgets=tuple(args.fore_iteration_budgets),
        )
        options = FOREFitOptions(
            batch_size=args.fore_batch_size,
            optimizer_steps=args.fore_optimizer_steps,
            target_action_draws=args.fore_target_action_draws,
            logit_clip=args.fore_logit_clip,
            grad_clip_norm=args.fore_grad_clip_norm,
            device=args.fore_device,
        )
        save_json(selection_manifest(candidates, options), output_dir / "selection_manifest.json")

    if args.mode == "example2-method-pilot":
        run_example2_method_pilot(output_dir, args)
        return
    if args.mode == "example1b-policy-audit":
        run_example1b_policy_audit_mode(output_dir, args)
        return
    if args.mode == "example1a-policy-audit":
        run_example1a_policy_audit_mode(output_dir, args)
        return
    if args.mode == "bias-acceptance":
        run_bias_acceptance_mode(output_dir, args)
        return
    if args.mode == "example2-nuisance-audit":
        run_example2_nuisance_audit_mode(output_dir, args)
        return
    if args.mode == "example2-paper-comparison":
        run_example2_paper_comparison_mode(output_dir, args)
        return
    if args.mode == "example2-semi-oracle":
        run_example2_semi_oracle(output_dir, args)
        return
    if args.mode == "example2-smoke":
        run_example2_smoke(output_dir, args)
        return
    if args.mode == "example2-shakedown":
        run_example2_shakedown_mode(output_dir, args)
        return
    if args.mode == "fore-selection-pilot":
        run_fore_selection_pilot_mode(output_dir, args)
        return
    if args.mode in {
        "data-fusion-smoke",
        "data-fusion-pilot",
        "data-fusion-confirmatory",
    }:
        run_data_fusion_mode(output_dir, args)
        return

    oracle = JRSSBOracle(build_config(args))
    if args.mode == "checks":
        run_checks(oracle, output_dir, args)
        return
    if args.mode == "pilot":
        run_pilot(oracle, output_dir, args)
        return
    run_mc(oracle, output_dir, args)


if __name__ == "__main__":
    main()
