import pandas as pd
import joblib
import pickle
import os
from src.url_features import extract_url_features
from src.content_features import extract_content_features, content_feature_names

# Load scaler and model with correct filenames
scaler_path = 'models/scaler.pkl'
model_path = 'models/voting_model.pkl'

print("="*60)
print("Feature Count Diagnostic")
print("="*60)

# Load scaler
if not os.path.exists(scaler_path):
    print(f"❌ ERROR: {scaler_path} not found.")
else:
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded from {scaler_path}")
    print(f"   Scaler expects {scaler.n_features_in_} features")
    if hasattr(scaler, 'feature_names_in_'):
        print("   Scaler feature names:")
        for i, name in enumerate(scaler.feature_names_in_.tolist(), 1):
            print(f"     {i:2d}. {name}")
    else:
        print("   (Scaler does not store feature names)")

# Load model
if not os.path.exists(model_path):
    print(f"❌ ERROR: {model_path} not found.")
else:
    model = pickle.load(open(model_path, 'rb'))
    print(f"\n✅ Model loaded from {model_path}")
    print(f"   Model expects {model.n_features_in_} features")
    if hasattr(model, 'feature_names_in_'):
        print("   Model feature names:")
        for i, name in enumerate(model.feature_names_in_.tolist(), 1):
            print(f"     {i:2d}. {name}")
    else:
        print("   (Model does not store feature names)")

# Test extraction on a sample URL
url = 'http://google.com'
print(f"\n🔍 Extracting features from {url}...")

# Extract URL features
df = pd.DataFrame({'url': [url]})
df = extract_url_features(df)

# Extract content features
content_feats = extract_content_features(url)
content_available = any(v != 0 for v in content_feats.values())

# Add content features to DataFrame
for key, value in content_feats.items():
    df[key] = value

print(f"✅ Extraction complete. DataFrame shape: {df.shape}")

# Define the feature lists used in app.py
url_feature_cols = [
    'url_length', 'dots', 'hyphens', 'at_symbol', 'question_marks', 'equals',
    'slashes', 'https', 'ip', 'suspicious_words', 'subdomains', 'domain_length',
    'path_length', 'query_length', 'num_digits', 'num_letters', 'special_chars',
    'has_port', 'double_slash', 'url_entropy', 'encoded_chars', 'num_parameters',
    'tld_length', 'https_count', 'digit_letter_ratio'
]
content_feature_cols = content_feature_names()  # this should match what was used in training
all_feature_cols = url_feature_cols + content_feature_cols

print(f"\n📋 Defined feature count in app: {len(all_feature_cols)}")
print(f"📊 Actual extracted feature count (excluding 'url'): {len(df.columns) - 1}")

df_cols = set(df.columns) - {'url'}
defined_set = set(all_feature_cols)

extra_in_df = df_cols - defined_set
missing_from_df = defined_set - df_cols

if extra_in_df:
    print("\n⚠️ Columns present in extracted data but NOT in defined list:")
    for col in extra_in_df:
        print(f"   - {col}")

if missing_from_df:
    print("\n⚠️ Columns in defined list but NOT in extracted data:")
    for col in missing_from_df:
        print(f"   - {col}")

if not extra_in_df and not missing_from_df:
    print("\n✅ All defined features are present in extracted data (and no extras).")

# If scaler has feature names, compare with defined list
if hasattr(scaler, 'feature_names_in_'):
    scaler_features = set(scaler.feature_names_in_)
    defined_set = set(all_feature_cols)
    extra_in_defined = defined_set - scaler_features
    missing_from_defined = scaler_features - defined_set

    print("\n" + "="*60)
    print("Comparison with Scaler's expected features")
    print("="*60)

    if extra_in_defined:
        print("\n⚠️ Features in defined list that scaler DOES NOT expect:")
        for col in extra_in_defined:
            print(f"   - {col}")

    if missing_from_defined:
        print("\n⚠️ Features scaler expects that are NOT in defined list:")
        for col in missing_from_defined:
            print(f"   - {col}")

    if not extra_in_defined and not missing_from_defined:
        print("\n✅ Defined list perfectly matches scaler's expected features!")

print("\n" + "="*60)
print("Diagnostic complete.")