# Experiment Log: kaggle-customer-churn-2026

> Track ALL experiments — submitted AND failed. This is the full history.
> AI: update this file after EVERY experiment, not just submissions.

## Experiments

| # | Date | Type | What Changed | CV Score | CV Std | LB Score | Submitted? | Commit | Seed | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-03-27 | baseline | LightGBM, 27 features, original merged, 5-fold | 0.9151 | ~0.001 | TBD | Pending | — | 42 | First model. Top features: AvgSpend, TotalCharges, MonthlyCharges |
| 2 | 2026-03-27 | ensemble | LGB+XGB+CatBoost, 40+ features, freq encoding | TBD | TBD | TBD | Pending | — | 42 | Training in progress |

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
- **Best CV:** Experiment #1 — LightGBM baseline, OOF AUC 0.9151
- **Best LB:** TBD — no submissions yet

## Selected Final Submissions
- **Final 1:** Experiment #[N] — [reasoning: conservative/aggressive, CV trust level]
- **Final 2:** Experiment #[N] — [reasoning]
