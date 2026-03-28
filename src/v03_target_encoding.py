"""
v03: Target Encoding + Multi-seed averaging
- Target encoding with K-fold regularization
- Multi-seed training (5 seeds per model)
- All v02 features included
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
N_SEEDS = 5  # Multi-seed averaging
USE_ORIGINAL = True
VERSION = "v03_target_enc_multiseed"

DATA_DIR = "data-kaggle"
ORIGINAL_PATH = "data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SUBMISSION_DIR = "submissions"

# =============================================================================
# DATA LOADING (same as v02)
# =============================================================================
print(f"Pipeline: {VERSION}")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

if USE_ORIGINAL and os.path.exists(ORIGINAL_PATH):
    orig = pd.read_csv(ORIGINAL_PATH)
    if orig['Churn'].dtype == object:
        orig['Churn'] = orig['Churn'].map({'Yes': 1, 'No': 0})
    if 'customerID' in orig.columns:
        orig = orig.drop('customerID', axis=1)
    if orig['TotalCharges'].dtype == object:
        orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    common_cols = [c for c in train.columns if c in orig.columns]
    if len(common_cols) > 5:
        orig_aligned = orig[common_cols].copy()
        for col in train.columns:
            if col not in orig_aligned.columns:
                orig_aligned[col] = np.nan
        orig_aligned = orig_aligned[train.columns]
        train = pd.concat([train, orig_aligned], ignore_index=True)

test_ids = test['id'].copy()
train = train.drop('id', axis=1, errors='ignore')
test = test.drop('id', axis=1, errors='ignore')
y = train['Churn'].map(lambda x: 1 if str(x) in ('Yes', '1', '1.0') else 0).astype(int)
train = train.drop('Churn', axis=1)

print(f"Train: {train.shape} | Test: {test.shape}")

# =============================================================================
# FEATURE ENGINEERING (from v02 + target encoding)
# =============================================================================
def add_all_features(df):
    df = df.copy()
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Ratios
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        df['TenureProxy'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)
        df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 0.01)
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['ChargesTenure'] = df['MonthlyCharges'] * df['tenure']
        df['ChargesTenureLog'] = np.log1p(df['MonthlyCharges'] * df['tenure'])

    # Service counts
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing = [c for c in service_cols if c in df.columns]
    df['NumServices'] = sum((df[c].astype(str) == 'Yes').astype(int) for c in existing)

    inet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_inet = [c for c in inet_services if c in df.columns]
    df['NumInternetServices'] = sum((df[c].astype(str) == 'Yes').astype(int) for c in existing_inet)
    df['NumSecurityServices'] = sum((df[c].astype(str) == 'Yes').astype(int)
                                     for c in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
                                     if c in df.columns)
    df['NumStreaming'] = sum((df[c].astype(str) == 'Yes').astype(int)
                             for c in ['StreamingTV', 'StreamingMovies'] if c in df.columns)

    # Bins
    if 'tenure' in df.columns:
        df['TenureBin'] = pd.cut(df['tenure'], bins=[-1, 6, 12, 24, 48, 72, 999],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)
    if 'MonthlyCharges' in df.columns:
        df['ChargesBin'] = pd.cut(df['MonthlyCharges'], bins=[-1, 20, 40, 60, 80, 100, 999],
                                   labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # Flags
    if 'InternetService' in df.columns:
        df['HasInternet'] = (df['InternetService'].astype(str) != 'No').astype(int)
        df['HasFiber'] = (df['InternetService'].astype(str) == 'Fiber optic').astype(int)
    if 'Contract' in df.columns:
        df['IsMonthToMonth'] = (df['Contract'].astype(str) == 'Month-to-month').astype(int)
    if 'PaymentMethod' in df.columns:
        df['IsElectronicCheck'] = (df['PaymentMethod'].astype(str) == 'Electronic check').astype(int)

    # Interactions
    pairs = [
        ('tenure', 'MonthlyCharges', 'tenure_x_monthly', 'mul'),
        ('IsMonthToMonth', 'tenure', 'MTM_x_tenure', 'mul'),
        ('HasFiber', 'MonthlyCharges', 'Fiber_x_charges', 'mul'),
        ('IsElectronicCheck', 'IsMonthToMonth', 'ECheck_x_MTM', 'mul'),
        ('NumSecurityServices', 'HasFiber', 'Security_x_Fiber', 'mul'),
        ('SeniorCitizen', 'IsMonthToMonth', 'Senior_x_MTM', 'mul'),
    ]
    for c1, c2, name, op in pairs:
        if c1 in df.columns and c2 in df.columns:
            df[name] = df[c1] * df[c2]

    return df

train = add_all_features(train)
test = add_all_features(test)

# Target encoding with K-fold regularization
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

def target_encode_kfold(train_df, test_df, col, target, n_splits=5, seed=42, smoothing=10):
    """Target encoding with K-fold to prevent leakage."""
    global_mean = target.mean()
    train_encoded = pd.Series(np.nan, index=train_df.index)
    test_encoded = pd.Series(0.0, index=test_df.index)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df, target):
        means = target.iloc[tr_idx].groupby(train_df[col].iloc[tr_idx].astype(str)).agg(['mean', 'count'])
        smooth_mean = (means['mean'] * means['count'] + global_mean * smoothing) / (means['count'] + smoothing)
        train_encoded.iloc[val_idx] = train_df[col].iloc[val_idx].astype(str).map(smooth_mean)

    # For test, use full train
    means = target.groupby(train_df[col].astype(str)).agg(['mean', 'count'])
    smooth_mean = (means['mean'] * means['count'] + global_mean * smoothing) / (means['count'] + smoothing)
    test_encoded = test_df[col].astype(str).map(smooth_mean)

    return train_encoded.fillna(global_mean), test_encoded.fillna(global_mean)

print("Target encoding...")
for col in cat_cols:
    train[f'{col}_te'], test[f'{col}_te'] = target_encode_kfold(train, test, col, y)

# Frequency encoding
for col in cat_cols:
    freq = train[col].value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq).fillna(0)
    test[f'{col}_freq'] = test[col].map(freq).fillna(0)

# Label encode original cats
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train = train.fillna(-999)
test = test.fillna(-999)

print(f"Total features: {train.shape[1]}")

# =============================================================================
# MULTI-SEED TRAINING
# =============================================================================
seeds = [SEED + i * 1000 for i in range(N_SEEDS)]
all_test_preds = []

for seed_idx, seed in enumerate(seeds):
    print(f"\n{'='*60}")
    print(f"Seed {seed_idx + 1}/{N_SEEDS}: {seed}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    t_lgb = np.zeros(len(test))
    t_xgb = np.zeros(len(test))
    t_cat = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # LGB
        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            learning_rate=0.03, num_leaves=63, min_child_samples=30,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
            n_estimators=5000, random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = m.predict_proba(X_val)[:, 1]
        t_lgb += m.predict_proba(test)[:, 1] / N_FOLDS

        # XGB
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            learning_rate=0.03, max_depth=7, min_child_weight=10,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            n_estimators=5000, random_state=seed, tree_method='hist',
            n_jobs=-1, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=200)
        oof_xgb[val_idx] = m.predict_proba(X_val)[:, 1]
        t_xgb += m.predict_proba(test)[:, 1] / N_FOLDS

        # CatBoost
        m = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=7,
            l2_leaf_reg=5, random_seed=seed, verbose=0, eval_metric='AUC')
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200)
        oof_cat[val_idx] = m.predict_proba(X_val)[:, 1]
        t_cat += m.predict_proba(test)[:, 1] / N_FOLDS

        print(f"  Fold {fold+1} - LGB: {roc_auc_score(y_val, oof_lgb[val_idx]):.6f} | "
              f"XGB: {roc_auc_score(y_val, oof_xgb[val_idx]):.6f} | "
              f"CAT: {roc_auc_score(y_val, oof_cat[val_idx]):.6f}")

    auc_l = roc_auc_score(y, oof_lgb)
    auc_x = roc_auc_score(y, oof_xgb)
    auc_c = roc_auc_score(y, oof_cat)
    oof_e = (oof_lgb + oof_xgb + oof_cat) / 3
    auc_e = roc_auc_score(y, oof_e)
    print(f"Seed {seed} -> LGB: {auc_l:.6f} | XGB: {auc_x:.6f} | CAT: {auc_c:.6f} | ENS: {auc_e:.6f}")

    # Average of 3 models for this seed
    t_ens = (t_lgb + t_xgb + t_cat) / 3
    all_test_preds.append(t_ens)

# Average across seeds
test_final = np.mean(all_test_preds, axis=0)

sub = sample_sub.copy()
sub['Churn'] = test_final
os.makedirs(SUBMISSION_DIR, exist_ok=True)
sub.to_csv(f"{SUBMISSION_DIR}/{VERSION}.csv", index=False)

print(f"\nSaved: {SUBMISSION_DIR}/{VERSION}.csv")
print(f"Range: [{test_final.min():.4f}, {test_final.max():.4f}], Mean: {test_final.mean():.4f}")
print("DONE!")
