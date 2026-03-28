"""
v02: Advanced Feature Engineering + Tuned Ensemble
- More interaction features
- Target encoding
- Frequency encoding
- Groupby aggregations
- Better hyperparameters
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
N_FOLDS = 10  # More folds for stability
USE_ORIGINAL = True
VERSION = "v02_advanced_fe"

DATA_DIR = "data-kaggle"
ORIGINAL_PATH = "data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SUBMISSION_DIR = "submissions"

np.random.seed(SEED)

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 60)
print(f"Pipeline: {VERSION}")
print("=" * 60)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# Merge original
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
        print(f"Merged original: {train.shape}")

# Separate
test_ids = test['id'].copy()
train = train.drop('id', axis=1, errors='ignore')
test = test.drop('id', axis=1, errors='ignore')

y = train['Churn'].map(lambda x: 1 if str(x) in ('Yes', '1', '1.0') else 0).astype(int)
train = train.drop('Churn', axis=1)

print(f"Train: {train.shape} | Test: {test.shape} | Churn rate: {y.mean():.4f}")

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
def advanced_features(df):
    df = df.copy()

    # Fix types
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # --- RATIO FEATURES ---
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        df['TenureProxy'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)
        df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 0.01)
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['ChargesTenure'] = df['MonthlyCharges'] * df['tenure']
        df['ChargesTenureLog'] = np.log1p(df['MonthlyCharges'] * df['tenure'])

    # --- SERVICE COUNT ---
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing = [c for c in service_cols if c in df.columns]
    if existing:
        df['NumServices'] = 0
        for col in existing:
            df['NumServices'] += (df[col].astype(str) == 'Yes').astype(int)

    # Internet-dependent services
    inet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_inet = [c for c in inet_services if c in df.columns]
    if existing_inet:
        df['NumInternetServices'] = 0
        for col in existing_inet:
            df['NumInternetServices'] += (df[col].astype(str) == 'Yes').astype(int)
        df['HasNoInternetService'] = 0
        for col in existing_inet:
            df['HasNoInternetService'] += (df[col].astype(str) == 'No internet service').astype(int)

    # Streaming count
    stream_cols = ['StreamingTV', 'StreamingMovies']
    existing_stream = [c for c in stream_cols if c in df.columns]
    if existing_stream:
        df['NumStreaming'] = 0
        for col in existing_stream:
            df['NumStreaming'] += (df[col].astype(str) == 'Yes').astype(int)

    # Security services
    sec_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    existing_sec = [c for c in sec_cols if c in df.columns]
    if existing_sec:
        df['NumSecurityServices'] = 0
        for col in existing_sec:
            df['NumSecurityServices'] += (df[col].astype(str) == 'Yes').astype(int)

    # --- BINS ---
    if 'tenure' in df.columns:
        df['TenureBin'] = pd.cut(df['tenure'], bins=[-1, 6, 12, 24, 48, 72, 999],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)
    if 'MonthlyCharges' in df.columns:
        df['ChargesBin'] = pd.cut(df['MonthlyCharges'], bins=[-1, 20, 40, 60, 80, 100, 999],
                                   labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # --- BOOLEAN FLAGS ---
    if 'InternetService' in df.columns:
        df['HasInternet'] = (df['InternetService'].astype(str) != 'No').astype(int)
        df['HasFiber'] = (df['InternetService'].astype(str) == 'Fiber optic').astype(int)
        df['HasDSL'] = (df['InternetService'].astype(str) == 'DSL').astype(int)
    if 'Contract' in df.columns:
        df['IsMonthToMonth'] = (df['Contract'].astype(str) == 'Month-to-month').astype(int)
    if 'PaymentMethod' in df.columns:
        df['IsElectronicCheck'] = (df['PaymentMethod'].astype(str) == 'Electronic check').astype(int)

    # --- INTERACTIONS ---
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']
    if 'IsMonthToMonth' in df.columns and 'tenure' in df.columns:
        df['MTM_x_tenure'] = df['IsMonthToMonth'] * df['tenure']
    if 'HasFiber' in df.columns and 'MonthlyCharges' in df.columns:
        df['Fiber_x_charges'] = df['HasFiber'] * df['MonthlyCharges']
    if 'IsElectronicCheck' in df.columns and 'IsMonthToMonth' in df.columns:
        df['ECheck_x_MTM'] = df['IsElectronicCheck'] * df['IsMonthToMonth']
    if 'NumSecurityServices' in df.columns and 'HasFiber' in df.columns:
        df['Security_x_Fiber'] = df['NumSecurityServices'] * df['HasFiber']
    if 'SeniorCitizen' in df.columns and 'IsMonthToMonth' in df.columns:
        df['Senior_x_MTM'] = df['SeniorCitizen'] * df['IsMonthToMonth']

    return df


train = advanced_features(train)
test = advanced_features(test)

# Frequency encoding for categoricals
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    freq = train[col].value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq).fillna(0)
    test[f'{col}_freq'] = test[col].map(freq).fillna(0)

# Label encode
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
print(f"\nTotal features: {len(feature_cols)}")

# =============================================================================
# CV + TRAINING
# =============================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_lgb = np.zeros(len(test))
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': -1,
    'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 0.5, 'reg_lambda': 1.0, 'n_estimators': 5000,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 10,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 0.5, 'reg_lambda': 2.0, 'n_estimators': 5000,
    'random_state': SEED, 'tree_method': 'hist', 'n_jobs': -1, 'verbosity': 0,
}

cat_params = {
    'iterations': 5000, 'learning_rate': 0.03, 'depth': 7,
    'l2_leaf_reg': 5, 'random_seed': SEED, 'verbose': 0,
    'eval_metric': 'AUC', 'task_type': 'CPU',
}

print("\nTraining...")
for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = m.predict_proba(X_val)[:, 1]
    test_lgb += m.predict_proba(test)[:, 1] / N_FOLDS

    # XGBoost
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=200)
    oof_xgb[val_idx] = m.predict_proba(X_val)[:, 1]
    test_xgb += m.predict_proba(test)[:, 1] / N_FOLDS

    # CatBoost
    m = CatBoostClassifier(**cat_params)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200)
    oof_cat[val_idx] = m.predict_proba(X_val)[:, 1]
    test_cat += m.predict_proba(test)[:, 1] / N_FOLDS

    print(f"  LGB: {roc_auc_score(y_val, oof_lgb[val_idx]):.6f} | "
          f"XGB: {roc_auc_score(y_val, oof_xgb[val_idx]):.6f} | "
          f"CAT: {roc_auc_score(y_val, oof_cat[val_idx]):.6f}")

# =============================================================================
# RESULTS
# =============================================================================
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat)
oof_ens = (oof_lgb + oof_xgb + oof_cat) / 3
auc_ens = roc_auc_score(y, oof_ens)

print(f"\nLGB: {auc_lgb:.6f} | XGB: {auc_xgb:.6f} | CAT: {auc_cat:.6f} | ENS: {auc_ens:.6f}")

# Optimize weights
best_auc, best_w = 0, (1/3, 1/3, 1/3)
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05: continue
        auc = roc_auc_score(y, w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat)
        if auc > best_auc:
            best_auc, best_w = auc, (w1, w2, w3)

print(f"Optimized: w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}) AUC={best_auc:.6f}")

test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat

sub = sample_sub.copy()
sub['Churn'] = test_final
os.makedirs(SUBMISSION_DIR, exist_ok=True)
sub.to_csv(f"{SUBMISSION_DIR}/{VERSION}.csv", index=False)

results = {
    'version': VERSION, 'seed': SEED, 'n_folds': N_FOLDS,
    'n_features': len(feature_cols),
    'cv_lgb': round(auc_lgb, 6), 'cv_xgb': round(auc_xgb, 6),
    'cv_cat': round(auc_cat, 6), 'cv_ens': round(auc_ens, 6),
    'cv_optimized': round(best_auc, 6),
    'weights': {'lgb': round(best_w[0], 2), 'xgb': round(best_w[1], 2), 'cat': round(best_w[2], 2)},
    'timestamp': datetime.now().isoformat(),
}
with open(f"{SUBMISSION_DIR}/{VERSION}_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {SUBMISSION_DIR}/{VERSION}.csv")
print(f"Range: [{test_final.min():.4f}, {test_final.max():.4f}], Mean: {test_final.mean():.4f}")
print("DONE!")
