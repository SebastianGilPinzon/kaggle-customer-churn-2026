# Next Experiments — Ranked by Priority

### Experiment: v01 — Baseline 3-Model Ensemble
- **Hypothesis:** LightGBM + XGBoost + CatBoost with basic FE + original data will score ~0.86-0.88 AUC
- **Changes:** src/baseline.py — 5-fold CV, basic features (ratios, bins, service count)
- **Effort:** 10 min
- **Expected impact:** Establish baseline LB score
- **Risk:** Low — standard approach
- **Priority:** 1 (IN PROGRESS)

### Experiment: v02 — Advanced Feature Engineering
- **Hypothesis:** More features (interactions, frequency encoding, flags) + 10 folds + lower LR will gain +0.005-0.01
- **Changes:** src/v02_advanced_fe.py — 10 folds, 45+ features, frequency encoding
- **Effort:** 30 min
- **Expected impact:** +0.005-0.01 over baseline
- **Risk:** Low — more features almost always help with GBDT
- **Priority:** 2

### Experiment: v03 — Target Encoding + Multi-seed
- **Hypothesis:** Target encoding adds signal for high-cardinality cats; multi-seed reduces variance
- **Changes:** src/v03_target_encoding.py — K-fold target encoding, 5 seeds × 5 folds × 3 models
- **Effort:** 2h (compute-intensive)
- **Expected impact:** +0.002-0.005 over v02
- **Risk:** Medium — target encoding can overfit if not done carefully
- **Priority:** 3

### Experiment: v04 — Optuna Hyperparameter Tuning
- **Hypothesis:** Tuned hyperparams for each model can gain +0.003-0.008
- **Changes:** New script with Optuna trial for LGB, XGB, CAT separately
- **Effort:** 2-4h
- **Expected impact:** +0.003-0.008 over v02
- **Risk:** Low — standard practice
- **Priority:** 4

### Experiment: v05 — Stacking with Ridge Meta-learner
- **Hypothesis:** 2-level stack (base: 3 GBDTs → meta: Ridge) captures complementary signals
- **Changes:** New script — OOF predictions from v02/v03 as features for Ridge
- **Effort:** 1h
- **Expected impact:** +0.001-0.003 over best single-level ensemble
- **Risk:** Medium — stacking on already-similar GBDT models may not help much
- **Priority:** 5
