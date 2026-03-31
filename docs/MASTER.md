# Competition Master Documentation — Predict Customer Churn (PS S6E3)

> Single source of truth for everything learned, discovered, and decided.

## 1. Competition Overview
- **Name:** Playground Series - Season 6, Episode 3
- **Goal:** Predict the probability of customer churn
- **URL:** https://www.kaggle.com/competitions/playground-series-s6e3
- **Type:** File upload (CSV with probabilities)
- **Metric:** AUC-ROC (area under the ROC curve)
- **Deadline:** March 31, 2026 (11:59 PM UTC)
- **Submissions:** 5/day, select 2 final
- **Teams:** 3,651+ (8,329 entrants)
- **Prize:** Kaggle merchandise (swag)
- **External data:** Allowed (CC BY 4.0)

### Submission Format
```
id,Churn
594194,0.1
594195,0.3
594196,0.2
```

## 2. Data Structure

### Competition Data (Synthetic)
- **Source:** Deep learning model trained on IBM Telco Customer Churn dataset
- **Size:** ~116.57 MB total, 43 columns
- **Files:** train.csv, test.csv, sample_submission.csv
- **Train:** 594,194 rows × 21 cols (19 features + id + Churn)
- **Test:** 254,655 rows × 20 cols
- **Churn rate:** 22.5% (imbalanced)
- **No nulls in competition data** (TotalCharges already float)

### Original Dataset (External — ALLOWED & DOWNLOADED)
- **Name:** IBM Telco Customer Churn
- **Location:** data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv
- **Rows:** 7,043
- **Columns:** 21 (20 features + Churn target)

### Features (from original)
| Feature | Type | Values |
|---------|------|--------|
| customerID | ID | unique |
| gender | Cat | Male/Female |
| SeniorCitizen | Binary | 0/1 |
| Partner | Cat | Yes/No |
| Dependents | Cat | Yes/No |
| tenure | Num | months with company |
| PhoneService | Cat | Yes/No |
| MultipleLines | Cat | Yes/No/No phone service |
| InternetService | Cat | DSL/Fiber optic/No |
| OnlineSecurity | Cat | Yes/No/No internet service |
| OnlineBackup | Cat | Yes/No/No internet service |
| DeviceProtection | Cat | Yes/No/No internet service |
| TechSupport | Cat | Yes/No/No internet service |
| StreamingTV | Cat | Yes/No/No internet service |
| StreamingMovies | Cat | Yes/No/No internet service |
| Contract | Cat | Month-to-month/One year/Two year |
| PaperlessBilling | Cat | Yes/No |
| PaymentMethod | Cat | Electronic check/Mailed check/Bank transfer/Credit card |
| MonthlyCharges | Num | monthly amount |
| TotalCharges | Num | total amount (may have blanks!) |
| Churn | Target | Yes/No (binary) |

### Most Important Features (SHAP from original)
1. Contract type (month-to-month = highest churn)
2. Tenure (shorter = higher churn)
3. MonthlyCharges (higher = more churn)
4. TechSupport (no = more churn)
5. InternetService (fiber optic = more churn)
6. OnlineSecurity (no = more churn)
7. PaymentMethod (electronic check = more churn)

## 3. Scoring System
- **Metric:** AUC-ROC
- **Higher = better** (1.0 = perfect, 0.5 = random)
- **Public/Private LB split** of test set

## 4. Score Progression
| # | Version | CV Score | LB Score | Key Change |
|---|---------|----------|----------|------------|
| 1 | v01_lgb_quick | 0.9151 | TBD | LightGBM, 27 features, 5-fold, original merged |
| 2 | v02_ensemble3 | 0.9159 | **0.91409** | LGB+XGB+CAT, 56 features, optimized weights (0.15/0.45/0.40) |
| 3 | v03_tuned_multiseed | ~0.9156 | 0.91401 | Tuned params + target enc + 3 seeds, 71 features. No improvement. |
| 4 | v04_exhaustive_fe | 0.9159 | **0.91426** | 134 features, groupby aggs, interactions. NEW BEST LB! CatBoost 50% weight. |
| 6 | v06_stacking | 0.9149 | 0.913 | 10-model stack + upweight 10x. WORSE — upweight hurts. |
| 7b | v07b_lite_stack | 0.9157 | 0.91392 | 7-GBDT stacking no upweight. Stacking didn't help. |
| 8 | v08_multiseed | ~0.9160 | 0.91427 | v04 features + 5-seed avg. |
| 9 | v09_no_original | 0.9168 | — | NO original data = best CV ever! |
| 10 | v10_woe_rank | 0.9160 | — | WoE + rank + clusters. No improvement. |
| 11 | v11_multiseed_no_orig | ~0.9168 | 0.91433 | v09 + 5 seeds. Original data hurts. |
| 12 | v12_blend_60 | — | 0.91431 | v11+v08 blend. No improvement. |
| 13 | v13_catboost_solo | 0.9162 | — | CatBoost only. Worse than ensemble. |
| 14 | v14_paradigm_shift | 0.9168 | 0.91423 | 12 models, 4 feature sets, hill climbing. |
| **15** | **v15_final_blend** | — | **0.91434** | **v11(70)+v14(30). BEST LB.** |
| 16 | v16_last_hope | — | 0.91432 | v11(75)+v08(25). No improvement. |

### Final Submissions Selected
- **Final 1:** v15 (LB 0.91434) — blend of diverse approaches
- **Final 2:** v11 (LB 0.91433) — pure multi-seed without original data

## 5. Strategy

### Execution Plan:
1. Baseline: LightGBM + StratifiedKFold 5 → submit for first LB
2. Add original dataset → submit
3. Feature engineering (interactions, ratios, groupby)
4. Train XGBoost + CatBoost → 3-model ensemble
5. Optuna tuning per model
6. Stacking with Ridge meta-learner
7. Multi-seed averaging
8. Pseudo-labeling (last resort)

### Feature Engineering Ideas:
- tenure × MonthlyCharges
- TotalCharges / tenure (effective monthly rate)
- TotalCharges / MonthlyCharges (tenure proxy)
- Number of services (count Yes across service columns)
- Contract × PaymentMethod interaction
- Tenure bins, MonthlyCharges bins
- Groupby means (charges by contract, internet service)
- Target encoding, frequency encoding

## 6. What Doesn't Work
*Updated as we learn*

## 7. Investigation Findings

### 2026-03-27 — Initial Research (5 parallel agents)
- Dataset is synthetic from IBM Telco Customer Churn
- Original dataset downloaded to data-original/
- PS competitions won by ensembles (50-75 models for gold)
- Minimum viable: XGBoost + LightGBM + CatBoost
- AutoGluon won 7 gold in 2024 PS
- CatBoost alone won PS S4E10
- All models MUST share same CV folds for stacking
- External data allowed — original ~7K rows help
- Class imbalance expected: ~26% churn / ~74% no churn
