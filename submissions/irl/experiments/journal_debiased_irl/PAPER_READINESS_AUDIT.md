# JASA experiment readiness audit

Status: complete and provenance verified. All nine predeclared confirmatory
cells contain 300 unique successful replications, and the strict assembler
recomputes their summaries from the raw rows. The manuscript tables use only
these locked v2 results.

## Locked reader-facing design

- Five-fold outer cross-fitting with normal influence-function intervals.
- Quadratic-logit logging policy and a fixed degree-2 multinomial sieve for the
  main study. This transparent primary design has good overlap; the original
  soft-MDP logging design is an appendix stress test.
- Neural FORE is the default for ordinary and signed advantage-weighted ratios.
  Architecture and optimizer are frozen from the truth-blind A-PBV pilot;
  fold-specific stopping is chosen from 30, 100, and 300 outer iterations.
  Neural optimization uses at most 20,000 deterministic training rows per fold;
  behavior-policy fitting and outer-fold inference still use all assigned rows.
- Example 1a uses n = 2,500, 5,000, and 10,000. Example 1b uses n = 25,000,
  50,000, and 100,000 because its influence function is materially noisier.
- Data fusion uses a transition log with known logging propensities and a
  separate abundant outcome-summary source. Inference is conditional on the
  frozen outcome regression.

## Locked confirmatory results

| Study | n | D-IRL bias | MC SD | Mean SE | 95% coverage | Ratio failures |
|---|---:|---:|---:|---:|---:|---:|
| Example 1a | 2,500 | 0.0074 | 0.1383 | 0.1270 | 0.933 | 0 |
| Example 1a | 5,000 | 0.0099 | 0.0911 | 0.0870 | 0.937 | 0 |
| Example 1a | 10,000 | 0.0049 | 0.0576 | 0.0608 | 0.963 | 0 |
| Example 1b | 25,000 | -0.0054 | 0.0193 | 0.0188 | 0.927 | 0 |
| Example 1b | 50,000 | -0.0025 | 0.0133 | 0.0132 | 0.953 | 0 |
| Example 1b | 100,000 | -0.0011 | 0.0094 | 0.0094 | 0.933 | 0 |
| Data fusion | 2,500 | 0.0029 | 0.0754 | 0.0728 | 0.947 | 0 |
| Data fusion | 5,000 | 0.0020 | 0.0504 | 0.0509 | 0.960 | 0 |
| Data fusion | 10,000 | -0.0008 | 0.0389 | 0.0358 | 0.937 | 0 |

Every cell's binomial Wilson 95% interval contains the nominal 0.95 coverage
level. The MC-SD/mean-SE ratio ranges from 0.95 to 1.09. All 2,700 rows have
finite inferential telemetry, zero ratio failures, and zero log-ratio clipping.
The provenance-checked paper artifacts are in
`artifacts/paper_confirmatory_v2_tables/`; its manifest pins all three input
runs and the five generated table/audit outputs by SHA-256.

## Bottleneck diagnosis

1. The original soft-MDP logging policy was the main problem. It produced
   ordinary ratio 99th percentiles near 24 and an effective-sample fraction near
   0.71. The primary quadratic design reduces these to about 12.4 and 0.93.
2. The decisive Example 1b bottleneck was a mathematical implementation bug,
   not sample size or A-PBV. Successor-initialized Jordan FORE estimates the
   strictly future signed resolvent, but the full Riesz representer also
   contains the current term
   (d^star(a,s){q^star(a,s)-V^star(s)}). The old implementation omitted
   that term. Its per-observation SD was 1.27, which almost exactly explains the
   old 16--18% gap between Monte Carlo SD and mean influence-function SE.
3. The estimator, oracle diagnostics, theorem, explicit DML formula, and proof
   now use the total current-plus-future signed representer and reconstruct its
   state average across all actions. A finite-state adjoint solve, the
   γ=0 edge case, and a reward finite-difference derivative test lock this
   convention. The former future-only code fails the derivative test.
4. FORE is not the remaining primary bottleneck. In paired corrected neural
   and exact-ratio pilot rows, estimator RMSE differences are 0.00228, 0.00165,
   and 0.00118 at n=25,000, 50,000, and 100,000. Ratio mass is 1.000--1.001,
   ESS is about 93%, and failures and clipping are zero.
5. Repeated cross-fitting and target-temperature changes did not reliably fix
   the stress design. They are not promoted to paper defaults.
6. The original one-SE logging-policy selector was too permissive because it
   used the raw best-model loss variance. The corrected selector uses paired
   per-observation excess losses. The primary design avoids residual selection
   noise by fixing the correctly specified quadratic sieve before confirmation.
7. A-PBV is operationally useful but the architecture pilot is not decisive:
   the selected 64-by-64, learning-rate 0.0003, weight-decay 0.001 candidate won
   30 of 160 events, while two alternatives won 28 each. This is acceptable
   because the library is deliberately narrow and the resulting ratios match
   exact-ratio diagnostics closely; it is not evidence that one architecture is
   intrinsically superior. In the locked study, A-PBV never selects 30 steps.
   The 300-step shares are 22%, 40%, and 64% across Example 1a; 77--79% across
   Example 1b; and 31%, 59%, and 79% across data fusion. Signed positive and
   negative shares are 80--82% and 69--71%, respectively. The cap is retained
   because the selected estimates are already close to the exact-ratio
   benchmark, not because the boundary selection is ignored.
8. Full-data FORE is needlessly expensive at the largest sample size because
   each outer fixed-point iteration reevaluates the complete ratio-training
   source. In four common n = 100,000 seeds, the 20,000-row cap changes the
   full-data FORE estimate by RMSE 0.00170 and the exact-ratio estimate by RMSE
   0.00142 (maximum 0.00217). Mean SE is 0.00763 capped versus 0.00761 full;
   ESS remains about 92,700/100,000; clipping and failures are zero. Median
   component fit telemetry improves by about 3.4-fold. This truth-blind
   computational budget is therefore the locked default; zero requests the
   appendix full-data diagnostic.

The true-policy diagnosis is seed-matched and uses exact adaptive ratios on
both sides. At n = 25,000 (20 common replications), the estimated-policy stress
run has bias -0.0204, MC SD 0.0418, mean SE 0.0340, and coverage 0.80; the
true-policy run has bias 0.0007, MC SD 0.0358, mean SE 0.0337, and coverage
0.90. At n = 10,000, true-policy substitution reduces absolute bias from
0.0291 to 0.0107. Thus the policy nuisance, not FORE, explains the principal
stress-design degradation.

## Corrected exact-ratio diagnostic pilot (50 replications per cell)

These rows isolate all non-ratio nuisances. They support the locked sample-size
grids but are not the reported neural-FORE experiment.

| Example | n | IF bias | MC SD | Mean SE | 95% coverage | Ratio failures |
|---|---:|---:|---:|---:|---:|---:|
| 1a | 2,500 | 0.0172 | 0.1351 | 0.1240 | 0.94 | 0 |
| 1a | 5,000 | 0.0094 | 0.0778 | 0.0855 | 0.96 | 0 |
| 1a | 10,000 | 0.0054 | 0.0646 | 0.0599 | 0.96 | 0 |
| 1b | 25,000 | -0.0056 | 0.0185 | 0.0189 | 0.98 | 0 |
| 1b | 50,000 | -0.0021 | 0.0135 | 0.0132 | 0.94 | 0 |
| 1b | 100,000 | -0.0044 | 0.0075 | 0.0093 | 0.96 | 0 |

The corrected Example 1b rows replace the obsolete future-only diagnostic.
They show the expected SE/SD agreement and near-nominal coverage without
changing folds, critical values, or tuning against estimand truth. The older
Example 1a rows are retained only as pre-v2 nuisance evidence; the v2 protocol
reruns all reader-facing cells under one source-locked identity.

The five-seed neural-FORE smoke for Example 1a has no failures. Paired
FORE-versus-exact estimate RMSE is 0.0265, 0.0175, and 0.0089 at n = 2,500,
5,000, and 10,000, respectively, equal to roughly 15--22% of the corresponding
mean SE. Its five-row coverage values are not used to assess coverage.

## Data-fusion pilot (20 replications per cell)

| n | IF bias | MC SD | Mean SE | 95% coverage | Ratio q99 | Ratio failures |
|---:|---:|---:|---:|---:|---:|---:|
| 2,500 | 0.0223 | 0.0755 | 0.0725 | 0.95 | 13.47 | 0 |
| 5,000 | -0.0101 | 0.0447 | 0.0513 | 1.00 | 13.79 | 0 |
| 10,000 | -0.0004 | 0.0336 | 0.0357 | 1.00 | 13.18 | 0 |

The frozen outcome regression has grid RMSE 0.00243 and induces an estimand
shift of 0.00301. This passes the locked gate of 0.00511, equal to 10% of the
pilot median standard error.

## Promotion outcome

The corrected protocol identity is `jasa-neural-fore-v2-corrected`. The strict
assembler accepts all nine cells: each has exactly 300 deterministic seeds,
finite inference and fold-level FORE telemetry, no ratio failure, pinned source
and nuisance hashes, and exact raw-to-summary agreement. Coverage is reported
for every predeclared cell and was not used to tune, select, or suppress a
result. No post-confirmatory method change was made.

`./run_paper_confirmatory.sh` reproduces the three run directories, and
`python assemble_paper_results.py` verifies them and writes the reader-facing
tables. The final manuscript points only to `paper_confirmatory_v2_tables`.

The directories `artifacts/main_confirmatory`,
`artifacts/data_fusion_confirmatory`, and all `paper_confirmatory_*` directories
without the `v2` marker are superseded. In particular, the completed old
Example 1a n=2,500 cell had 91.7% coverage (Wilson interval 88.0--94.3%) and is
not promoted. The v2 source-aware fingerprint and immutable cell identity make
those rows impossible to resume into the corrected study.
