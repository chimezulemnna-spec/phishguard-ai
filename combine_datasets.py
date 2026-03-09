"""
combine_datasets.py
───────────────────
Builds a balanced combined_urls.csv from phishing_urls.csv
and legitimate_urls.csv.

FIXES:
  1. legitimate_urls.csv has bare domains → add https://
  2. Old run produced 64207 phishing vs 10000 legit (86/14 split).
     Now we balance 1:1 up to min(n_phishing, n_legit).
  3. Drops duplicates before saving.
"""

import os
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, 'dataset')
OUT_FILE     = os.path.join(DATASET_DIR, 'combined_urls.csv')

print("=" * 60)
print("COMBINING & BALANCING PHISHING / LEGITIMATE DATASETS")
print("=" * 60)

# ── 1. Phishing ───────────────────────────────────────────────────────────────
phish_path = os.path.join(DATASET_DIR, 'phishing_urls.csv')
df_phish = pd.read_csv(phish_path)[['url']].copy()
df_phish['url'] = df_phish['url'].astype(str).str.strip()
df_phish = df_phish.dropna(subset=['url']).drop_duplicates(subset=['url'])
df_phish['label'] = 1
print(f"\n[1/3] Phishing URLs loaded:    {len(df_phish):>7,}")

# ── 2. Legitimate ─────────────────────────────────────────────────────────────
legit_path = os.path.join(DATASET_DIR, 'legitimate_urls.csv')
df_legit = pd.read_csv(legit_path)

# Normalise column name
if 'url' not in df_legit.columns:
    for candidate in ('URL', 'domain', 'domain_name'):
        if candidate in df_legit.columns:
            df_legit = df_legit.rename(columns={candidate: 'url'})
            break

df_legit = df_legit[['url']].copy()
df_legit['url'] = df_legit['url'].astype(str).str.strip()

# FIX: add protocol to bare domains
def ensure_protocol(u):
    return u if u.startswith(('http://', 'https://')) else 'https://' + u

df_legit['url'] = df_legit['url'].apply(ensure_protocol)
df_legit = df_legit.dropna(subset=['url']).drop_duplicates(subset=['url'])
df_legit['label'] = 0
print(f"[2/3] Legitimate URLs loaded:  {len(df_legit):>7,}")

# ── 3. Balance 1 : 1 ─────────────────────────────────────────────────────────
n = min(len(df_phish), len(df_legit))
df_phish  = df_phish.sample(n=n,  random_state=42)
df_legit  = df_legit.sample(n=n,  random_state=42)

combined = pd.concat([df_phish, df_legit], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_csv(OUT_FILE, index=False)

print(f"[3/3] Balanced dataset saved:  {len(combined):>7,}  →  {OUT_FILE}")
print(f"\n  Phishing  (label=1): {(combined['label']==1).sum():,}")
print(f"  Legitimate(label=0): {(combined['label']==0).sum():,}")
print("\n✅ Done.")