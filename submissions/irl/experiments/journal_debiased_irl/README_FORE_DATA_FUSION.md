# Neural FORE and data-fusion experiments

Run all commands from this directory with the RLtools virtual environment. The
paper-facing default is `--ratio-mode neural-fore`; `oracle` is reserved for
labeled diagnostics.

## Staged workflow

1. Operational smoke test (two replications per example):

   ```bash
   /Users/larsvanderlaan/repos/RLtools/.venv/bin/python run_jrssb_simulation.py \
     --mode fore-smoke --output-dir artifacts/fore_smoke
   ```

2. Truth-blind A-PBV architecture/optimizer pilot:

   ```bash
   /Users/larsvanderlaan/repos/RLtools/.venv/bin/python run_jrssb_simulation.py \
     --mode fore-selection-pilot --jobs 3 \
     --output-dir artifacts/fore_selection_pilot
   ```

   The locked configuration is written to
   `artifacts/fore_selection_pilot/frozen_fore_config.json`. Selection uses only
   held-out A-PBV scores. Simulation ratios and estimand truth are not arguments
   to the selector. The default ten seeds yield 160 selection events after
   crossing examples, endpoint sample sizes, outer folds, and signed components.

3. Locked main-study pilot, followed by the confirmatory run. The primary
   design uses the prespecified quadratic logging policy, a correctly specified
   quadratic multinomial sieve, five-fold cross-fitting, and a 41-by-41 oracle
   integration grid. Neural ratio optimization uses a deterministic maximum of
   20,000 outer-training rows per fold; policy fitting and inference use every
   assigned row. Example 1a uses sample sizes 2,500, 5,000, and 10,000;
   the more variable Example 1b uses 25,000, 50,000, and 100,000. The original
   soft-MDP logging policy is retained as an explicitly labeled stress design.

   ```bash
   /Users/larsvanderlaan/repos/RLtools/.venv/bin/python run_jrssb_simulation.py \
     --mode monte-carlo --repetitions 10 --examples 1a \
     --sample-sizes 2500 5000 10000 --crossfit-folds 5 --jobs 3 \
     --behavior-policy-design quadratic-logit --main-sieve-mode fixed-quadratic \
     --fore-frozen-config artifacts/fore_selection_pilot/frozen_fore_config.json \
     --output-dir artifacts/main_pilot_example1a

   /Users/larsvanderlaan/repos/RLtools/.venv/bin/python run_jrssb_simulation.py \
     --mode monte-carlo --repetitions 10 --examples 1b \
     --sample-sizes 25000 50000 100000 --crossfit-folds 5 --jobs 3 \
     --behavior-policy-design quadratic-logit --main-sieve-mode fixed-quadratic \
     --fore-frozen-config artifacts/fore_selection_pilot/frozen_fore_config.json \
     --output-dir artifacts/main_pilot_example1b
   ```

   The three locked 300-repetition stages are collected in one resumable
   script. It deliberately fixes the sample-size grids, seeds, nuisance
   families, A-PBV stopping budgets, and SHA-256 identities of frozen pilot
   objects. The corrected protocol writes only to `paper_confirmatory_v2_*`;
   earlier `paper_confirmatory_*` checkpoints are preserved but incompatible:

   ```bash
   ./run_paper_confirmatory.sh
   ```

4. Data-fusion smoke and locked pilot. The outcome regression is selected by
   held-out outcome MSE on a deterministic 80/20 split of the million-row
   source. Its selected 80%-sample fit is frozen and reused in every later
   stage. The transition source records its known logging propensity; the
   Gaussian transition sieve is selected within each outer training fold by
   held-out next-state MSE. Source independence is a simulation convenience,
   not an estimator requirement.

   ```bash
   /Users/larsvanderlaan/repos/RLtools/.venv/bin/python run_jrssb_simulation.py \
     --mode data-fusion-smoke --jobs 2 \
     --fore-frozen-config artifacts/fore_selection_pilot/frozen_fore_config.json \
     --output-dir artifacts/data_fusion_smoke_locked

   /Users/larsvanderlaan/repos/RLtools/.venv/bin/python run_jrssb_simulation.py \
     --mode data-fusion-pilot --jobs 3 \
     --fore-frozen-config artifacts/fore_selection_pilot/frozen_fore_config.json \
     --fusion-g-cache artifacts/data_fusion_summary_gamma80_v4_shared/frozen_outcome_regression.pkl \
     --output-dir artifacts/data_fusion_pilot_transition_locked
   ```

   The paper wrapper pins
   `artifacts/data_fusion_paper_pilot_v1/data_fusion_manifest.json`, derives
   `pilot_median_se` from that verified manifest, and enforces the predeclared
   10% induced-shift gate. Do not reconstruct the confirmatory command from an
   older smoke or pilot directory; use `./run_paper_confirmatory.sh`.

All long stages are resumable at the configuration-and-source-keyed cell level.
Each checkpoint has an immutable identity sidecar. Resume fails if its source,
configuration, frozen inputs, or cell identity differs. Each completed output
directory contains the run configuration, environment metadata,
machine-readable estimates and summaries, selection/ratio diagnostics,
reader-facing LaTeX tables, and a SHA-256 artifact manifest. The paper
assembler reads these run directories, verifies raw 300-row cells and their
lineage, recomputes summaries, and emits a diagnostic coverage-calibration
audit. It never selects or suppresses a completed cell using simulation truth.

## Interpretation guardrails

- A-PBV and all nuisance selectors use observed held-out losses only. Oracle
  ratios and estimand truth are diagnostic outputs and cannot enter selection.
- Exact-ratio runs are paired to neural-FORE runs with common replication
  seeds. Never compare differently seeded output directories as if they were a
  nuisance ablation.
- The 20,000-row neural optimization budget was promoted only after a paired
  four-seed n=100,000 audit: capped-versus-exact estimate RMSE 0.00142,
  maximum absolute difference 0.00217, zero failures/clipping, and about a
  3.4-fold reduction in per-component ratio-fit time.
- The fixed outcome regression passed the predeclared shift gate: absolute
  induced estimand shift 0.00301 versus the allowed 0.00511 (10% of the locked
  pilot median standard error).
- The data-fusion intervals are conditional on the abundant frozen outcome
  source. They do not claim to include joint uncertainty from estimating the
  outcome summary.
- Do not copy pilot or smoke coverage into the manuscript as confirmatory
  evidence. Promote tables only after all 300 rows per cell and the artifact
  manifest are present.

## Focused tests

```bash
/Users/larsvanderlaan/repos/RLtools/.venv/bin/python -m pytest tests -q
```
