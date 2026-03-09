import pandas as pd
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from src.url_features import extract_url_features
from src.content_features import extract_content_features, content_feature_names

# ===================== CONFIGURATION =====================
DATASET_PATH = os.path.join('dataset', 'combined_urls.csv')
MODEL_SAVE_PATH = os.path.join('models', 'voting_model.pkl')
SCALER_SAVE_PATH = os.path.join('models', 'scaler.pkl')
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42

# ===================== LOAD AND CLEAN DATASET =====================
print("=" * 60)
print("TRAINING COMBINED MODEL (URL + CONTENT FEATURES)")
print("=" * 60)

print(f"\nLoading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Remove duplicates
initial_count = len(df)
df = df.drop_duplicates(subset=['url'])
duplicates_removed = initial_count - len(df)
print(f"Removed {duplicates_removed} duplicate URLs")
print(f"Total unique URLs: {len(df)}")

# Check class balance
print("\nClass distribution:")
print(df['label'].value_counts())
phishing_count = len(df[df['label'] == 1])
legit_count = len(df[df['label'] == 0])
print(f"  Phishing (1): {phishing_count} URLs ({phishing_count/len(df)*100:.1f}%)")
print(f"  Legitimate (0): {legit_count} URLs ({legit_count/len(df)*100:.1f}%)")

# ===================== SPLIT DATA (BEFORE FEATURE EXTRACTION) =====================
print("\n[1/5] Splitting data into train/test sets...")
train_df, test_df = train_test_split(
    df, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=df['label']  # Maintain class balance
)

print(f"  Training set: {len(train_df)} URLs")
print(f"  Test set: {len(test_df)} URLs")

# ===================== FEATURE COLUMNS =====================
url_feature_cols = [
    'url_length', 'dots', 'hyphens', 'at_symbol',
    'question_marks', 'equals', 'slashes', 'https',
    'ip', 'suspicious_words', 'subdomains', 'domain_length',
    'path_length', 'query_length', 'num_digits', 'num_letters',
    'special_chars', 'has_port', 'double_slash', 'url_entropy',
    'encoded_chars', 'num_parameters', 'tld_length',
    'https_count', 'digit_letter_ratio'
]
content_feature_cols = content_feature_names()  # Now includes 'content_fetched'
all_feature_cols = url_feature_cols + content_feature_cols

# ===================== EXTRACT URL FEATURES (FAST) =====================
print("\n[2/5] Extracting URL features...")
start = time.time()
train_url_feats = extract_url_features(train_df[['url']])
test_url_feats = extract_url_features(test_df[['url']])
print(f"  ✓ URL features extracted in {time.time()-start:.2f} seconds")

# ===================== EXTRACT CONTENT FEATURES (SLOW) =====================
print("\n[3/5] Extracting content features (this will take time)...")

# Check for cached features
cache_file = os.path.join('dataset', 'content_features_cache.pkl')
use_cache = True

if use_cache and os.path.exists(cache_file):
    print("  Loading cached content features...")
    cache_data = pd.read_pickle(cache_file)
    
    # Check if cache matches current data
    if len(cache_data) == len(df):
        print("  Cache matches dataset size. Using cached features.")
        # Split cached features according to train/test indices
        train_content_df = cache_data.iloc[train_df.index].reset_index(drop=True)
        test_content_df = cache_data.iloc[test_df.index].reset_index(drop=True)
    else:
        print("  Cache size mismatch. Extracting fresh features...")
        use_cache = False
else:
    use_cache = False

if not use_cache:
    # Extract for training set
    print("  Processing training URLs...")
    train_content_list = []
    train_failed = 0
    for i, url in enumerate(train_df['url']):
        if i % 500 == 0:
            print(f"    Train: {i}/{len(train_df)} (failed: {train_failed})")
        feats = extract_content_features(url)
        train_content_list.append(feats)
        if feats.get('content_fetched', 0) == 0:
            train_failed += 1
    
    # Extract for test set
    print("\n  Processing test URLs...")
    test_content_list = []
    test_failed = 0
    for i, url in enumerate(test_df['url']):
        if i % 500 == 0:
            print(f"    Test: {i}/{len(test_df)} (failed: {test_failed})")
        feats = extract_content_features(url)
        test_content_list.append(feats)
        if feats.get('content_fetched', 0) == 0:
            test_failed += 1
    
    train_content_df = pd.DataFrame(train_content_list)
    test_content_df = pd.DataFrame(test_content_list)
    
    # Cache the full content features
    full_content_df = pd.concat([train_content_df, test_content_df], ignore_index=True)
    full_content_df.to_pickle(cache_file)
    print(f"\n  Content features cached to {cache_file}")

# Calculate fetch statistics
train_fetched = train_content_df['content_fetched'].sum()
train_fetch_rate = (train_fetched / len(train_content_df)) * 100
test_fetched = test_content_df['content_fetched'].sum()
test_fetch_rate = (test_fetched / len(test_content_df)) * 100

print(f"\n  Content fetch statistics:")
print(f"    Training: {train_fetched}/{len(train_content_df)} URLs fetched ({train_fetch_rate:.1f}%)")
print(f"    Test: {test_fetched}/{len(test_content_df)} URLs fetched ({test_fetch_rate:.1f}%)")

# ===================== COMBINE FEATURES =====================
print("\n[4/5] Combining features...")

X_train = pd.concat([train_url_feats[url_feature_cols], train_content_df[content_feature_cols]], axis=1)
X_test = pd.concat([test_url_feats[url_feature_cols], test_content_df[content_feature_cols]], axis=1)
y_train = train_df['label']
y_test = test_df['label']

# Fill any remaining NaN values (shouldn't happen, but just in case)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"  Training features shape: {X_train.shape}")
print(f"  Test features shape: {X_test.shape}")

# ===================== SCALE FEATURES =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== TRAIN MODEL =====================
print("\n[5/5] Training Voting Classifier (RF + XGB)...")
start = time.time()

# Use class weights if dataset is imbalanced
if phishing_count != legit_count:
    print(f"  Using class weights to handle imbalance")
    class_weight = 'balanced'
else:
    class_weight = None

rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=RANDOM_STATE, 
    n_jobs=-1,
    class_weight=class_weight
)

xgb = XGBClassifier(
    n_estimators=100, 
    random_state=RANDOM_STATE, 
    n_jobs=-1, 
    eval_metric='logloss',
    scale_pos_weight=legit_count/phishing_count if phishing_count > 0 else 1
)

model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
model.fit(X_train_scaled, y_train)

print(f"  ✓ Training completed in {time.time()-start:.2f} seconds")

# ===================== EVALUATE MODEL =====================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Training accuracy
train_pred = model.predict(X_train_scaled)
train_acc = (train_pred == y_train).mean()
print(f"\nTraining accuracy: {train_acc:.4f}")

# Test accuracy (THIS IS THE ONE THAT MATTERS!)
test_pred = model.predict(X_test_scaled)
test_acc = (test_pred == y_test).mean()
print(f"Test accuracy: {test_acc:.4f}")

# Check for overfitting
if train_acc - test_acc > 0.05:
    print("\n⚠️  WARNING: Large gap between train and test accuracy!")
    print("   The model may be overfitting. Consider:")
    print("   - Reducing model complexity")
    print("   - Getting more training data")
    print("   - Adding regularization")
elif test_acc > 0.95:
    print("\n✅ GREAT! Your model is performing well on unseen data!")
else:
    print("\n📊 Model performance is okay. Can be improved with more features or data.")

# Calculate fetch rate impact
print("\n📈 Fetch Rate Analysis:")
print(f"   URLs with content fetched: {test_fetched}/{len(test_df)} ({test_fetch_rate:.1f}%)")

if test_fetch_rate < 50:
    print("   ⚠️  Low fetch rate - content features may not be helping much")
    print("   Consider improving fetch_page function or using a headless browser")
elif test_fetch_rate > 80:
    print("   ✅ Good fetch rate - content features should be effective")

# ===================== SAVE MODEL & SCALER =====================
os.makedirs('models', exist_ok=True)
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"✅ Scaler saved to: {SCALER_SAVE_PATH}")

# ===================== SAVE FEATURE NAMES FOR REFERENCE =====================
feature_names_file = os.path.join('models', 'feature_names.txt')
with open(feature_names_file, 'w') as f:
    f.write("Feature columns used in model:\n")
    f.write("-" * 40 + "\n")
    for i, col in enumerate(all_feature_cols):
        f.write(f"{i+1:2d}. {col}\n")

print(f"✅ Feature names saved to: {feature_names_file}")

# ===================== QUICK TEST ON SAMPLE URLS =====================
print("\n" + "=" * 60)
print("QUICK TEST ON SAMPLE URLs")
print("=" * 60)

sample_urls = [
    'https://www.google.com',
    'https://www.paypal.com',
    'https://www.facebook.com',
    'http://paypal-verify-account.com',  # Suspicious
    'https://apple-id-login.net',         # Suspicious
]

for url in sample_urls:
    # Create a single-row DataFrame
    df_sample = pd.DataFrame({'url': [url]})
    
    # Extract features
    url_feats_sample = extract_url_features(df_sample)
    content_feats_sample = extract_content_features(url)
    
    # Combine
    for key, value in content_feats_sample.items():
        url_feats_sample[key] = value
    
    # Select and scale features
    features_sample = url_feats_sample[all_feature_cols].fillna(0)
    features_scaled = scaler.transform(features_sample.values)
    
    # Predict
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    result = "⚠️ PHISHING" if pred == 1 else "✅ LEGITIMATE"
    confidence = max(prob) * 100
    
    fetched = content_feats_sample.get('content_fetched', 0)
    fetch_status = "Yes" if fetched == 1 else "No"
    
    print(f"\nURL: {url}")
    print(f"  Result: {result}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Content fetched: {fetch_status}")