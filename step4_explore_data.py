# step4_explore_data_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('src')

print("="*60)
print("Exploratory Data Analysis")
print("="*60)

# Load the data
df = pd.read_csv('PILWD_features.csv')

# 1. Basic dataset info
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📋 Column Names:")
for i, col in enumerate(df.columns[:20]):  # Show first 20
    print(f"  {i+1:2d}. {col}")
if len(df.columns) > 20:
    print(f"  ... and {len(df.columns)-20} more")

# 2. Target distribution
print("\n🎯 Target Distribution:")
print(df['label'].value_counts())
print(f"\n  Legitimate (0): {df['label'].value_counts()[0]:,} ({df['label'].value_counts()[0]/len(df)*100:.2f}%)")
print(f"  Phishing (1):   {df['label'].value_counts()[1]:,} ({df['label'].value_counts()[1]/len(df)*100:.2f}%)")

# 3. Feature correlations (top positive and negative)
print("\n🔗 Top Correlations with Phishing Label")
print("-" * 50)

correlations = df.corr()['label'].drop('label').sort_values(ascending=False)
# Remove any NaN values
correlations = correlations.dropna()

print("\n📈 Top 10 POSITIVE correlations (features that INCREASE with phishing):")
for feat, corr in correlations.head(10).items():
    category = '🌐 URL' if feat.startswith('U') else '📄 Content' if feat.startswith('H') else '📊 Other'
    print(f"  {category} | {feat}: {corr:.4f}")

print("\n📉 Top 10 NEGATIVE correlations (features that DECREASE with phishing):")
for feat, corr in correlations.tail(10).items():
    category = '🌐 URL' if feat.startswith('U') else '📄 Content' if feat.startswith('H') else '📊 Other'
    print(f"  {category} | {feat}: {corr:.4f}")

# 4. Feature importance by category
print("\n📊 Feature Distribution by Category:")
print("-" * 50)

url_features = [col for col in df.columns if col.startswith('U') and col != 'label']
content_features = [col for col in df.columns if col.startswith('H')]
other_features = [col for col in df.columns if col.startswith(('Y', 'N')) and col != 'label']

print(f"  🌐 URL Features: {len(url_features)}")
print(f"  📄 Content Features: {len(content_features)}")
print(f"  📊 Other Features: {len(other_features)}")

# 5. Summary statistics for key features
print("\n📈 Key Feature Statistics:")
print("-" * 50)
key_features = ['U_Special_Ratio', 'H_Script_Count', 'H_Hidden_Count', 
                'H_Iframe_Count', 'NLP_Urgent_Count', 'H_Ext_Ratio']

for feature in key_features:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(f"  Legitimate (0) - mean: {df[df['label']==0][feature].mean():.4f}, std: {df[df['label']==0][feature].std():.4f}")
        print(f"  Phishing (1)   - mean: {df[df['label']==1][feature].mean():.4f}, std: {df[df['label']==1][feature].std():.4f}")
        print(f"  Difference: {((df[df['label']==1][feature].mean() - df[df['label']==0][feature].mean()) / df[df['label']==0][feature].mean() * 100):+.2f}%")

# 6. Check for any missing values
print("\n🔍 Missing Values Check:")
missing = df.isnull().sum().sum()
if missing == 0:
    print("  ✅ No missing values found!")
else:
    print(f"  ⚠️ Found {missing} missing values")

# 7. Create visualization
try:
    os.makedirs('assets', exist_ok=True)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    # Plot top 6 features
    top_features = correlations.head(6).index.tolist()
    
    for i, feature in enumerate(top_features):
        # Clip outliers for better visualization
        q99 = df[feature].quantile(0.99)
        data_legit = df[df['label']==0][feature].clip(upper=q99)
        data_phish = df[df['label']==1][feature].clip(upper=q99)
        
        axes[i].hist(data_legit, bins=30, alpha=0.6, label='Legitimate', density=True, color='blue')
        axes[i].hist(data_phish, bins=30, alpha=0.6, label='Phishing', density=True, color='red')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].set_title(f'{feature}\n(corr: {correlations[feature]:.3f})')
    
    plt.tight_layout()
    plt.savefig('assets/pilwd_feature_analysis.png', dpi=100)
    print(f"\n✅ Visualization saved to assets/pilwd_feature_analysis.png")
    
except Exception as e:
    print(f"\n⚠️ Could not create visualization: {e}")

print("\n✅ Exploratory Analysis Complete!")