"""
Data Preparation Script
Cleans, combines, and prepares data for training
"""

import pandas as pd
from src.url_features import extract_url_features
import os

print("="*50)
print("🔧 PREPARING DATA")
print("="*50)

# ===================== STEP 1: LOAD AND CLEAN =====================
print("\n📥 Step 1: Loading datasets...")

# Load phishing data
df_phish = pd.read_csv('data/phishing_urls.csv')
print(f"   Phishing loaded: {len(df_phish)} rows")

# Load legitimate data  
df_legit = pd.read_csv('data/legitimate_urls.csv')
print(f"   Legitimate loaded: {len(df_legit)} rows")

# ===================== STEP 2: EXTRACT URL AND ADD LABELS =====================
print("\n🏷️ Step 2: Extracting URLs and adding labels...")

# Keep only url column + add label
df_phish = df_phish[['url']].copy()
df_phish['label'] = 1  # Phishing = 1
print(f"   Phishing: {len(df_phish)} URLs with label=1")

df_legit = df_legit[['url']].copy()
df_legit['label'] = 0  # Legitimate = 0
print(f"   Legitimate: {len(df_legit)} URLs with label=0")

# ===================== STEP 3: FIX LEGITIMATE URLs =====================
print("\n🔗 Step 3: Adding https:// to legitimate URLs...")

# Check if URLs already have protocol
def has_protocol(url):
    return str(url).startswith('http://') or str(url).startswith('https://')

# Add https:// if missing
df_legit['url'] = df_legit['url'].apply(
    lambda x: x if has_protocol(x) else 'https://' + x
)

print(f"   Fixed legitimate URLs")

# ===================== STEP 4: BALANCE DATASET =====================
print("\n⚖️ Step 4: Balancing dataset...")

# Use equal numbers (or you can use more legitimate)
# Let's use 100k from each for faster training
n_samples = min(100000, len(df_phish), len(df_legit))

df_phish_sample = df_phish.sample(n=n_samples, random_state=42)
df_legit_sample = df_legit.sample(n=n_samples, random_state=42)

print(f"   Using {n_samples} from each class")

# Combine
df = pd.concat([df_phish_sample, df_legit_sample], ignore_index=True)
print(f"   Total combined: {len(df)} URLs")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===================== STEP 5: EXTRACT FEATURES =====================
print("\n🔍 Step 5: Extracting URL features...")

# This adds: url_length, dots, hyphens, at_symbol, https, ip, suspicious_words
df = extract_url_features(df)

print(f"   Features extracted: {df.columns.tolist()}")

# ===================== STEP 6: SAVE CLEANED DATA =====================
print("\n💾 Step 6: Saving cleaned data...")

# Keep only feature columns + label
feature_cols = ['url_length', 'dots', 'hyphens', 'at_symbol', 'https', 'ip', 'suspicious_words', 'label']
df_clean = df[feature_cols]

# Save
df_clean.to_csv('data/combined_clean.csv', index=False)

print(f"   ✅ Saved {len(df_clean)} rows to data/combined_clean.csv")

# Show summary
print("\n" + "="*50)
print("📊 DATA SUMMARY")
print("="*50)
print(f"\nTotal samples: {len(df_clean)}")
print(f"Phishing (label=1): {(df_clean['label']==1).sum()}")
print(f"Legitimate (label=0): {(df_clean['label']==0).sum()}")
print(f"\nFeature columns: {feature_cols}")
print(f"\nClass distribution:")
print(df_clean['label'].value_counts())