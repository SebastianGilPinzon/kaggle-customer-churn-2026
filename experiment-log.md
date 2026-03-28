# Experiment Log: kaggle-customer-churn-2026

> Track ALL experiments — submitted AND failed. This is the full history.
> AI: update this file after EVERY experiment, not just submissions.

## Experiments

| # | Date | Type | What Changed | CV Score | CV Std | LB Score | Submitted? | Commit | Seed | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-03-27 | baseline | LightGBM, 27 features, original merged, 5-fold | 0.9151 | ~0.001 | TBD | Pending | — | 42 | First model. Top features: AvgSpend, TotalCharges, MonthlyCharges |
| 2 | 2026-03-27 | ensemble | LGB+XGB+CatBoost, 56 features, freq encoding | 0.9159 | ~0.001 | **0.91409** | Yes | 8332480 | 42 | LGB=0.9154, XGB=0.9157, CAT=0.9156. CV-LB gap: 0.002 |

### Column Guide
- **Type:** baseline / feature / model / tuning / ensemble / postprocess / bugfix
- **CV Std:** standard deviation across folds (high = unstable model)
- **Submitted?:** Yes / No (reason)
- **Commit:** git commit hash that produced this experiment
- **Seed:** random seed used

## CV-LB Correlation Tracker
| Submission # | CV Score | LB Score | CV-LB Gap | Trend |
|---|---|---|---|---|
| 1 | 0.XXX | 0.XXX | +0.XXX | — |

> If CV-LB gap changes direction for 3+ consecutive submissions → STOP and re-evaluate CV strategy.

## Current Best
- **Best CV:** Experiment #2 — 3-model ensemble, OOF AUC 0.9159
- **Best LB:** Experiment #2 — 3-model ensemble, LB 0.91409

## Selected Final Submissions
- **Final 1:** Experiment #[N] — [reasoning: conservative/aggressive, CV trust level]
- **Final 2:** Experiment #[N] — [reasoning]
