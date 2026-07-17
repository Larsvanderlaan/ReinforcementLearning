# Clipped KL-FORI research-readiness audit

This audit maps the mathematical and experimental contract to executable code.
It does not contain confirmatory evidence or empirical manuscript claims.

## Controlled baseline

- Starting RLtools revision: `1422757ad7f1de84893423251f113fae622bd9db`.
- Original dirty-tree content hash at isolation:
  `ebbd350e5a8853adbe5e83dcac642e479ae255f270cae1d604715a4dc4573f38`.
- Isolated branch: `codex/clipped-paper-ready`.
- CPU is the reproducibility reference; Torch determinism is requested and
  recorded for neural fits.

## Executable audit matrix

| Requirement | Implementation | Verification |
|---|---|---|
| Gate weights are `tau_upper/n`, `(1-gamma)/m`, and `gamma*omega/n`, without class balancing | `_clipped_kl_fori_objectives.py` | hand-computed objectives, replication, initial-weight, and gradient tests |
| Projection uses generalized KL with `E_nu exp(h)`, no log partition, no successor normalization, and no output normalization | objective specification and backend updates | objective, scaling, mass, and finite-difference tests |
| Hard gate uses deterministic ties and remains frozen during a ratio update | linear and neural backends | gate-freeze and tie tests |
| Fitted log ratio is smoothly envelope bounded and starts at exactly one | backend initializers | arbitrary-input linear and neural initialization tests |
| Population truth uses an independent analytic box-constrained oracle | `_clipped_coverage_oracle.py` | randomized differential and shared-hub formula tests |
| Outer stopping uses the maximum ratio-change and gate-change residual | backend outer loops | convergence, patience, and oscillating-gate tests |
| Validation records losses but does not tune; refit status is separate | fit orchestration | selection/refit history tests |
| Raw fit arrays are absent by default and model artifacts are versioned NPZ | public model facade | payload-retention, malformed-artifact, and round-trip tests |
| Standard KL-FORI remains normalized and its legacy public API remains unchanged | `kl_fori.py`, `_kl_fori_impl.py`, and package facade | 30 core standard tests plus the complete pre-existing package suite |
| `q=0` standard and post-hoc rows are `out_of_regime` | coverage execution | structured-status tests |
| Each method is scored against its own estimand | coverage metrics | explicit own-mass, own-value, and own-ratio tests |
| Empirical-support and population sampling errors sum to total clipped mass error | coverage metrics | exact decomposition test |
| Pilot selection is truth blind | explicit deployable-field allowlists | adversarial oracle-field deletion tests |
| Linear, standard, and neural optimizers are frozen separately | `clipped-coverage-freeze-v1` | backend routing, revision, schema, and grid tests |
| Parallel runs are atomic, resumable, complete, and duplicate-free | per-fold artifacts and strict shard merger | cell-ID, resume, sharding, incompatibility, and completeness tests |
| Results carry revision, dirty hash, environment, config hash, cell IDs, folds, and seeds | artifact manifests | smoke and manifest tests |

## Paper-ready workload

- Main linear shared-hub experiment: 2,500 dataset cells, 5,000 fold
  artifacts, and 7,500 method rows.
- Contextual appendix: 300 backend-specific dataset cells, 600 fold artifacts,
  and 300 clipped-method rows.
- Sensitivities: 540 dataset cells, 1,080 fold artifacts, and 1,140 method
  rows.
- Confirmatory and sensitivity execution requires a clean checkout whose
  revision and full grids match the committed freeze manifest.

## Verification snapshot

- Ruff passes for the complete package source and tests.
- All 107 clipped-estimator and clipped-coverage tests pass.
- All 30 isolated standard KL-FORI core tests pass. Five copied tests for a
  separate dirty-tree package-wide benchmark/API migration are explicitly
  skipped to preserve this worktree's isolation; the existing package suite
  passes unchanged.
- The complete occupancy-ratio benchmark suite passes.
- Combined clipped-estimator and clipped-coverage statement coverage is 91%,
  above the 90% gate; the main estimator and execution modules individually
  meet or exceed 90%.

## Remaining release gates

Do not launch confirmatory or sensitivity runs until:

1. the historical clipped-linear pilot reproduces exactly on the committed
   clean code revision, including fold prediction and gate hashes;
2. the truth-blind standard budget pilot selects a zero-failure budget;
3. the contextual neural screen and three-restart final select a zero-failure,
   stable candidate; and
4. the resulting `clipped-coverage-freeze-v1` manifest is committed outside
   the clean RLtools source worktree.

No oracle mass, ratio, or stopped-value error may enter these decisions.
