"""
train_models.py  –  Complete Training Pipeline
═══════════════════════════════════════════════
Trains a soft-voting ensemble (Random Forest + XGBoost) on 38 features
(25 URL + 13 content, excluding content_fetched from feature matrix).

FIXES applied vs. previous version
────────────────────────────────────
1. Dataset was 86 % phishing / 14 % legitimate.
   → combine_datasets.py must be run first to produce a balanced file.
     If it still isn't balanced, we auto-downsample here.

2. content_features_cache.pkl had doubled column names
   ('content_content_num_forms' etc.) because an old extraction loop
   double-prefixed the keys.  The new content_features.py is consistent;
   the cache is rebuilt if needed.

3. scaler.pkl was trained on 38 features but rf_model.pkl had 25 features.
   Now ALL models (rf, xgb, voting) are trained on the same 38-feature
   scaled matrix; scaler and models are always in sync.

4. ensemble.py used voting='hard' → changed to 'soft' so predict_proba
   and ROC-AUC work correctly.

5. train_test_split is done BEFORE feature extraction so there is no
   data leakage from the scaler.
"""

import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, PROJECT_ROOT)

from src.url_features      import extract_url_features, URL_FEATURE_COLS
from src.content_features  import (extract_content_features,
                                   content_feature_names, CONTENT_FEATURE_COLS)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  xgboost not installed – using GradientBoostingClassifier instead")
    from sklearn.ensemble import GradientBoostingClassifier

# ── config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = os.path.join(PROJECT_ROOT, 'dataset', 'combined_urls.csv')
CACHE_PATH    = os.path.join(PROJECT_ROOT, 'dataset', 'content_features_cache.pkl')
MODELS_DIR    = os.path.join(PROJECT_ROOT, 'models')
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# Features fed to the model (content_fetched is a flag, not a signal)
CONTENT_MODEL_COLS = [c for c in CONTENT_FEATURE_COLS if c != 'content_fetched']
ALL_FEATURE_COLS   = URL_FEATURE_COLS + CONTENT_MODEL_COLS   # 38 total

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  PHISHING DETECTION — FULL TRAINING PIPELINE")
print("=" * 60)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset …")
df = pd.read_csv(DATASET_PATH)
df['url'] = df['url'].astype(str).str.strip()
df = df.dropna(subset=['url']).drop_duplicates(subset=['url'])

label_counts = df['label'].value_counts()
print(f"  Total: {len(df):,}   phishing={label_counts.get(1,0):,}   "
      f"legitimate={label_counts.get(0,0):,}")

# Auto-balance if still skewed
n_min = label_counts.min()
if label_counts.max() / n_min > 1.5:
    print(f"  ⚠️  Dataset imbalanced – downsampling to {n_min:,} per class")
    df = pd.concat([
        df[df['label'] == 0].sample(n_min, random_state=RANDOM_STATE),
        df[df['label'] == 1].sample(n_min, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Balanced: {len(df):,} total")

# ── 2. Train/test split ───────────────────────────────────────────────────────
print("\n[2/6] Splitting data …")
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label']
)
print(f"  Train: {len(train_df):,}   Test: {len(test_df):,}")

# ── 3. URL features ───────────────────────────────────────────────────────────
print("\n[3/6] Extracting URL features …")
t0 = time.time()
train_url = extract_url_features(train_df[['url']].copy())
test_url  = extract_url_features(test_df[['url']].copy())
print(f"  ✓ Done in {time.time()-t0:.1f}s  ({len(URL_FEATURE_COLS)} features)")

# ── 4. Content features ───────────────────────────────────────────────────────
print("\n[4/6] Loading / building content features …")

def _valid_cache(path, expected_len):
    """Return cache DataFrame only if it's valid (correct columns, right size)."""
    if not os.path.exists(path):
        return None
    try:
        cache = pd.read_pickle(path)
        # Must have the right columns (no doubled prefix, right length)
        has_doubled = any(c.startswith('content_content') for c in cache.columns)
        if has_doubled:
            print("  ⚠️  Cache has doubled column names – rebuilding …")
            return None
        if len(cache) != expected_len:
            print(f"  ⚠️  Cache length {len(cache)} ≠ {expected_len} – rebuilding …")
            return None
        missing = [c for c in CONTENT_FEATURE_COLS if c not in cache.columns]
        if missing:
            print(f"  ⚠️  Cache missing columns {missing} – rebuilding …")
            return None
        return cache
    except Exception as e:
        print(f"  ⚠️  Cache unreadable ({e}) – rebuilding …")
        return None

cache = _valid_cache(CACHE_PATH, len(df))

if cache is not None:
    print(f"  ✓ Valid cache found  ({len(cache):,} rows)")
    # Align cache rows by URL — safe regardless of shuffle order
    train_content = cache.loc[train_df['url'].values].reset_index(drop=True)
    test_content  = cache.loc[test_df['url'].values].reset_index(drop=True)
else:
    print("  Skipping content fetch for training (URL features only — sites offline)")
    zero_row = {c: 0 for c in CONTENT_FEATURE_COLS}
    train_content = pd.DataFrame([zero_row] * len(train_df))
    test_content  = pd.DataFrame([zero_row] * len(test_df))

    # Save cache keyed by URL
    full_content = pd.concat([train_content, test_content], ignore_index=True)
    full_content['url'] = df['url'].reset_index(drop=True)
    full_content = full_content.set_index('url')
    full_content.to_pickle(CACHE_PATH)
    print(f"  ✓ Cache saved → {CACHE_PATH}")

# ── 5. Assemble feature matrices ──────────────────────────────────────────────
print("\n[5/6] Assembling feature matrices …")

X_train = pd.concat(
    [train_url[URL_FEATURE_COLS].reset_index(drop=True),
     train_content[CONTENT_MODEL_COLS].reset_index(drop=True)],
    axis=1
).fillna(0)

X_test = pd.concat(
    [test_url[URL_FEATURE_COLS].reset_index(drop=True),
     test_content[CONTENT_MODEL_COLS].reset_index(drop=True)],
    axis=1
).fillna(0)

y_train = train_df['label'].reset_index(drop=True)
y_test  = test_df['label'].reset_index(drop=True)

print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"  Features: {ALL_FEATURE_COLS}")

# Scale (fit only on train to avoid leakage)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Scaler fitted on {scaler.n_features_in_} features ✓")

# ── 6. Train models ───────────────────────────────────────────────────────────
print("\n[6/6] Training models …")

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=5,
    random_state=RANDOM_STATE, n_jobs=-1
)

if HAS_XGB:
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    booster = XGBClassifier(
        n_estimators=100, max_depth=8, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=n0 / n1 if n1 > 0 else 1
    )
    booster_name = 'xgb'
else:
    from sklearn.ensemble import GradientBoostingClassifier
    booster = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    booster_name = 'gb'

# Voting ensemble (soft so predict_proba works)
voting = VotingClassifier(
    estimators=[('rf', rf), (booster_name, booster)],
    voting='soft'
)

for name, model in [('Random Forest', rf),
                    (booster_name.upper(), booster),
                    ('Voting Ensemble', voting)]:
    t0 = time.time()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    print(f"\n  {name}  ({time.time()-t0:.1f}s)")
    print(f"    Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"    F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"    Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"    Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"    ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# ── Save everything ───────────────────────────────────────────────────────────
print("\n── Saving models ──────────────────────────────────────────")
with open(os.path.join(MODELS_DIR, 'rf_model.pkl'),      'wb') as f: pickle.dump(rf,     f)
with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'),     'wb') as f: pickle.dump(booster, f)
with open(os.path.join(MODELS_DIR, 'voting_model.pkl'),  'wb') as f: pickle.dump(voting,  f)
with open(os.path.join(MODELS_DIR, 'scaler.pkl'),        'wb') as f: pickle.dump(scaler,  f)

# Save feature column list for app.py to use
with open(os.path.join(MODELS_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(ALL_FEATURE_COLS))

print(f"  ✓ rf_model.pkl      – {rf.n_features_in_} features")
print(f"  ✓ xgb_model.pkl     – {booster.n_features_in_} features")
print(f"  ✓ voting_model.pkl  – {voting.n_features_in_} features")
print(f"  ✓ scaler.pkl        – {scaler.n_features_in_} features")
print(f"  ✓ feature_names.txt – {len(ALL_FEATURE_COLS)} feature names")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE ✅")
print("=" * 60)