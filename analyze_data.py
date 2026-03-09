import pandas as pd

print("="*50)
print("📊 ANALYZING TRAINING DATA")
print("="*50)

# Load datasets
df_phish = pd.read_csv('data/phishing_urls.csv')
df_legit = pd.read_csv('data/legitimate_urls.csv')

print(f"\n✅ Phishing: {len(df_phish)} rows")
print(f"   Columns: {df_phish.columns.tolist()}")
print(df_phish.head(3))

print(f"\n✅ Legitimate: {len(df_legit)} rows")
print(f"   Columns: {df_legit.columns.tolist()}")
print(df_legit.head(3))