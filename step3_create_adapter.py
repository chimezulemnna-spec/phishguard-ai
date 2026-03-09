# step3_create_adapter.py
import pandas as pd
import numpy as np
import os
import joblib
import sys
sys.path.append('src')

print("="*60)
print("Creating Data Adapter for PILWD Dataset")
print("="*60)

class PhishingDatasetAdapter:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.feature_names = None
        self.url_features = []
        self.content_features = []
        self.other_features = []
        
    def load_and_analyze(self):
        print("\n📊 Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        
        # Categorize features
        all_cols = self.df.columns[:-1]  # exclude label
        
        self.url_features = [col for col in all_cols if col.startswith('U')]
        self.content_features = [col for col in all_cols if col.startswith('H')]
        self.other_features = [col for col in all_cols 
                              if col.startswith('Y') or col.startswith('N')]
        
        self.feature_names = all_cols.tolist()
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total samples: {len(self.df):,}")
        print(f"  Total features: {len(self.feature_names)}")
        print(f"  URL features: {len(self.url_features)}")
        print(f"  Content features: {len(self.content_features)}")
        print(f"  Other features: {len(self.other_features)}")
        
        # Target distribution
        phishing_count = self.df['label'].sum()
        legitimate_count = len(self.df) - phishing_count
        phishing_pct = (phishing_count / len(self.df)) * 100
        
        print(f"\n🎯 Target Distribution:")
        print(f"  Legitimate (0): {legitimate_count:,} ({100-phishing_pct:.2f}%)")
        print(f"  Phishing (1):   {phishing_count:,} ({phishing_pct:.2f}%)")
        
        return self.df
    
    def get_feature_groups(self):
        return {
            'url_features': self.url_features,
            'content_features': self.content_features,
            'other_features': self.other_features,
            'all_features': self.feature_names
        }

# Test the adapter
adapter = PhishingDatasetAdapter('PILWD_features.csv')
df = adapter.load_and_analyze()

# Show sample of the data
print(f"\n🔍 First 5 rows of data (first 10 columns):")
print(df.iloc[:5, :10])

# Show feature groups
groups = adapter.get_feature_groups()
print(f"\n📁 Feature Groups:")
for group_name, features in groups.items():
    if features:
        print(f"  {group_name}: {len(features)} features")
        if len(features) > 0:
            print(f"    Examples: {features[:3]}")

# Save feature mapping
os.makedirs('models', exist_ok=True)
mapping = {
    'feature_names': adapter.feature_names,
    'url_features': adapter.url_features,
    'content_features': adapter.content_features,
    'other_features': adapter.other_features
}
joblib.dump(mapping, 'models/feature_mapping_pilwd.pkl')
print(f"\n✅ Feature mapping saved to models/feature_mapping_pilwd.pkl")