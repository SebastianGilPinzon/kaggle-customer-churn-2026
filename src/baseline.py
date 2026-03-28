"""
Baseline pipeline for Predict Customer Churn (PS S6E3)
- LightGBM + XGBoost + CatBoost ensemble
- StratifiedKFold CV
- Optional: append original dataset
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
N_FOLDS = 5
USE_ORIGINAL = True  # Append original Telco dataset
VERSION = "v01_baseline_ensemble"

DATA_DIR = "data-kaggle"
ORIGINAL_PATH = "data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SUBMISSION_DIR = "submissions"

np.random.seed(SEED)

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 60)
print(f"Pipeline: {VERSION}")
print(f"Seed: {SEED} | Folds: {N_FOLDS} | Original: {USE_ORIGINAL}")
print("=" * 60)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"\nTrain shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample sub:  {sample_sub.shape}")
print(f"\nTarget distribution:\n{train['Churn'].value_counts(normalize=True)}")

# =============================================================================
# MERGE ORIGINAL DATASET
# =============================================================================
if USE_ORIGINAL and os.path.exists(ORIGINAL_PATH):
    orig = pd.read_csv(ORIGINAL_PATH)
    print(f"\nOriginal dataset: {orig.shape}")

    # Map Churn Yes/No to 1/0 if needed
    if orig['Churn'].dtype == object:
        orig['Churn'] = orig['Churn'].map({'Yes': 1, 'No': 0})

    # Drop customerID
    if 'customerID' in orig.columns:
        orig = orig.drop('customerID', axis=1)

    # Fix TotalCharges (string with blanks in original)
    if orig['TotalCharges'].dtype == object:
        orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')

    # Align columns with train (only keep matching columns)
    common_cols = [c for c in train.columns if c in orig.columns]
    print(f"Common columns: {len(common_cols)}/{len(train.columns)}")

    if len(common_cols) > 5:  # Only merge if enough columns match
        orig_aligned = orig[common_cols].copy()
        # Add missing columns as NaN
        for col in train.columns:
            if col not in orig_aligned.columns:
                orig_aligned[col] = np.nan
        orig_aligned = orig_aligned[train.columns]  # Reorder
        train = pd.concat([train, orig_aligned], ignore_index=True)
        print(f"After merge: {train.shape}")
    else:
        print(f"WARNING: Only {len(common_cols)} common columns, skipping original merge")

# =============================================================================
# PREPROCESSING
# =============================================================================
# Separate target and IDs
if 'id' in train.columns:
    train_ids = train['id']
    train = train.drop('id', axis=1)
else:
    train_ids = None

test_ids = test['id'] if 'id' in test.columns else test.index
if 'id' in test.columns:
    test = test.drop('id', axis=1)

y = train['Churn'].copy()
train = train.drop('Churn', axis=1)

# Convert target to int (handles mixed Yes/No and 0/1 from original merge)
y = y.map(lambda x: 1 if str(x) in ('Yes', '1', '1.0') else 0).astype(int)

print(f"\nFeatures: {train.shape[1]}")
print(f"Target: {y.value_counts().to_dict()}")

# Identify column types
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = train.select_dtypes(include=['number']).columns.tolist()
print(f"Categorical: {len(cat_cols)} | Numeric: {len(num_cols)}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def add_features(df):
    """Add engineered features."""
    df = df.copy()

    # Fix TotalCharges if string
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Ratio features
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        df['TenureProxy'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['ChargesTenure'] = df['MonthlyCharges'] * df['tenure']

    # Service count
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_service_cols = [c for c in service_cols if c in df.columns]
    if existing_service_cols:
        df['NumServices'] = 0
        for col in existing_service_cols:
            if df[col].dtype in ('object', 'string', 'str'):
                df['NumServices'] += (df[col].astype(str) == 'Yes').astype(int)
            else:
                df['NumServices'] += (df[col] > 0).astype(int)

    # Tenure bins
    if 'tenure' in df.columns:
        df['TenureBin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, 999],
                                  labels=[0, 1, 2, 3, 4]).astype(float)

    # MonthlyCharges bins
    if 'MonthlyCharges' in df.columns:
        df['ChargesBin'] = pd.cut(df['MonthlyCharges'], bins=[0, 30, 50, 70, 90, 999],
                                   labels=[0, 1, 2, 3, 4]).astype(float)

    # Has internet service
    if 'InternetService' in df.columns:
        df['HasInternet'] = (df['InternetService'].astype(str) != 'No').astype(int)

    return df

train = add_features(train)
test = add_features(test)

# Update column lists after FE
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = train.select_dtypes(include=['number']).columns.tolist()
all_features = cat_cols + num_cols

print(f"\nAfter FE: {train.shape[1]} features ({len(cat_cols)} cat, {len(num_cols)} num)")

# =============================================================================
# LABEL ENCODE CATEGORICALS
# =============================================================================
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Fill NaN
train = train.fillna(-999)
test = test.fillna(-999)

feature_cols = train.columns.tolist()
print(f"Final features: {len(feature_cols)}")

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Storage for OOF predictions
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

# Storage for test predictions
test_lgb = np.zeros(len(test))
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

# LightGBM params
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 2000,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

# XGBoost params
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 2000,
    'random_state': SEED,
    'tree_method': 'hist',
    'n_jobs': -1,
    'verbosity': 0,
}

# CatBoost params
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': SEED,
    'verbose': 0,
    'eval_metric': 'AUC',
    'task_type': 'CPU',
}

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(test)[:, 1] / N_FOLDS
    lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=100
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(test)[:, 1] / N_FOLDS
    xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(test)[:, 1] / N_FOLDS
    cat_auc = roc_auc_score(y_val, oof_cat[val_idx])

    print(f"  LGB: {lgb_auc:.6f} | XGB: {xgb_auc:.6f} | CAT: {cat_auc:.6f}")

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat)

print(f"LightGBM  OOF AUC: {auc_lgb:.6f}")
print(f"XGBoost   OOF AUC: {auc_xgb:.6f}")
print(f"CatBoost  OOF AUC: {auc_cat:.6f}")

# Simple average ensemble
oof_ens = (oof_lgb + oof_xgb + oof_cat) / 3
test_ens = (test_lgb + test_xgb + test_cat) / 3
auc_ens = roc_auc_score(y, oof_ens)
print(f"\nEnsemble  OOF AUC: {auc_ens:.6f}")

# Optimized weights (grid search)
best_auc = 0
best_w = (1/3, 1/3, 1/3)
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)

print(f"\nOptimized weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}")
print(f"Optimized AUC:    {best_auc:.6f}")

# Final predictions with optimized weights
test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat

# =============================================================================
# SUBMISSION
# =============================================================================
sub = sample_sub.copy()
sub['Churn'] = test_final
sub.to_csv(f"{SUBMISSION_DIR}/{VERSION}.csv", index=False)
print(f"\nSubmission saved: {SUBMISSION_DIR}/{VERSION}.csv")
print(f"Shape: {sub.shape}")
print(f"Predictions range: [{test_final.min():.4f}, {test_final.max():.4f}]")
print(f"Mean prediction: {test_final.mean():.4f}")

# Save results summary
results = {
    'version': VERSION,
    'seed': SEED,
    'n_folds': N_FOLDS,
    'use_original': USE_ORIGINAL,
    'n_features': len(feature_cols),
    'train_shape': list(train.shape),
    'test_shape': list(test.shape),
    'cv_lgb': round(auc_lgb, 6),
    'cv_xgb': round(auc_xgb, 6),
    'cv_cat': round(auc_cat, 6),
    'cv_ensemble': round(auc_ens, 6),
    'cv_optimized': round(best_auc, 6),
    'weights': {'lgb': round(best_w[0], 2), 'xgb': round(best_w[1], 2), 'cat': round(best_w[2], 2)},
    'timestamp': datetime.now().isoformat(),
}
with open(f"{SUBMISSION_DIR}/{VERSION}_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {SUBMISSION_DIR}/{VERSION}_results.json")
print("\nDONE!")
