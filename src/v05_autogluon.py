"""v05: AutoGluon — historically dominates Playground Series."""
import pandas as pd, numpy as np, sys, os, json, warnings
warnings.filterwarnings('ignore')

VERSION = "v05_autogluon"
SEED = 42
TIME_LIMIT = 3600  # 1 hour

print('Loading...', file=sys.stderr, flush=True)
train = pd.read_csv('data-kaggle/train.csv')
test = pd.read_csv('data-kaggle/test.csv')
sub = pd.read_csv('data-kaggle/sample_submission.csv')

# Merge original
orig = pd.read_csv('data-original/WA_Fn-UseC_-Telco-Customer-Churn.csv')
orig['Churn'] = orig['Churn'].map({'Yes': 'Yes', 'No': 'No'})  # Keep as string for AG
orig.drop('customerID', axis=1, inplace=True)
orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
common = [c for c in train.columns if c in orig.columns]
oa = orig[common].copy()
for c in train.columns:
    if c not in oa.columns: oa[c] = np.nan
train = pd.concat([train, oa[train.columns]], ignore_index=True)

test_ids = test['id'].copy()
train.drop('id', axis=1, errors='ignore', inplace=True)
test.drop('id', axis=1, errors='ignore', inplace=True)

# Basic FE (AutoGluon handles categoricals natively)
for df in [train, test]:
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['AvgSpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenProxy'] = df['TotalCharges'] / (df['MonthlyCharges'] + .01)
    df['ChTen'] = df['MonthlyCharges'] * df['tenure']
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + .01)
    svc = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['nSvc'] = sum((df[c].astype(str) == 'Yes').astype(int) for c in svc)
    df['Fiber'] = (df['InternetService'].astype(str) == 'Fiber optic').astype(int)
    df['MTM'] = (df['Contract'].astype(str) == 'Month-to-month').astype(int)
    df['EChk'] = (df['PaymentMethod'].astype(str) == 'Electronic check').astype(int)
    df['ExpectedTotal'] = df['tenure'] * df['MonthlyCharges']
    df['TotalDiff'] = df['TotalCharges'] - df['ExpectedTotal']

print(f'Train: {train.shape}, Test: {test.shape}', file=sys.stderr, flush=True)

# AutoGluon
from autogluon.tabular import TabularPredictor

label = 'Churn'
predictor = TabularPredictor(
    label=label,
    eval_metric='roc_auc',
    path='autogluon_models',
    problem_type='binary',
).fit(
    train,
    time_limit=TIME_LIMIT,
    presets='best_quality',
    num_cpus=os.cpu_count(),
    verbosity=2,
)

# Leaderboard
lb = predictor.leaderboard(silent=True)
print('\nLeaderboard:', file=sys.stderr, flush=True)
print(lb.to_string(), file=sys.stderr, flush=True)

# Predict
preds = predictor.predict_proba(test)
# Get probability of churn (Yes)
if 'Yes' in preds.columns:
    test_pred = preds['Yes'].values
elif 1 in preds.columns:
    test_pred = preds[1].values
else:
    test_pred = preds.iloc[:, 1].values

os.makedirs('submissions', exist_ok=True)
sub['Churn'] = test_pred
sub.to_csv(f'submissions/{VERSION}.csv', index=False)

# Best model info
best = predictor.get_model_best()
val_score = predictor.evaluate(train)
print(f'\nBest model: {best}', file=sys.stderr, flush=True)
print(f'Val score: {val_score}', file=sys.stderr, flush=True)

json.dump({'version': VERSION, 'best_model': str(best), 'time_limit': TIME_LIMIT,
           'feats': train.shape[1]},
          open(f'submissions/{VERSION}_results.json', 'w'), indent=2)

print(f'\nSaved: submissions/{VERSION}.csv', file=sys.stderr, flush=True)
print('DONE!', file=sys.stderr, flush=True)
