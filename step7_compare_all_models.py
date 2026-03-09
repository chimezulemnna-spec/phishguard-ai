# step7_compare_all_models.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

print("="*60)
print("Comparing All Models")
print("="*60)

# Load test data
df = pd.read_csv('PILWD_features.csv')
feature_cols = [col for col in df.columns if col != 'label']
X = df[feature_cols]
y = df['label']

# Load scaler
scaler = joblib.load('models/scaler_pilwd.pkl')
X_scaled = scaler.transform(X)

# Load all models
models = {
    'Random Forest (PILWD)': joblib.load('models/random_forest_pilwd.pkl'),
    'XGBoost (PILWD)': joblib.load('models/xgboost_pilwd.pkl'),
    'Gradient Boosting (PILWD)': joblib.load('models/gradient_boosting_pilwd.pkl'),
    'Voting Ensemble (PILWD)': joblib.load('models/voting_ensemble_pilwd.pkl')
}

# Add your existing models if they exist
existing_paths = {
    'Random Forest (Existing)': 'models/rf_model.pkl',
    'XGBoost (Existing)': 'models/xgb_model.pkl',
    'Voting (Existing)': 'models/voting_model.pkl'
}

for name, path in existing_paths.items():
    if os.path.exists(path):
        try:
            models[name] = joblib.load(path)
            print(f"✅ Loaded {name}")
        except:
            print(f"⚠️ Could not load {name}")

# Evaluate all models
results = []
for name, model in models.items():
    try:
        if 'Voting' in name and 'PILWD' in name:
            # Voting ensemble needs predict_proba
            y_proba = model.predict_proba(X_scaled)[:, 1]
        elif 'PILWD' in name:
            y_proba = model.predict_proba(X_scaled)[:, 1]
        else:
            # Existing models - need to handle feature mismatch
            print(f"⚠️ Skipping {name} - needs feature alignment")
            continue
        
        y_pred = (y_proba > 0.5).astype(int)
        
        results.append({
            'Model': name,
            'ROC-AUC': roc_auc_score(y, y_proba),
            'F1-Score': f1_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred)
        })
        print(f"✅ Evaluated {name}")
    except Exception as e:
        print(f"⚠️ Error evaluating {name}: {e}")

# Create comparison dataframe
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:")
print(results_df.to_string(index=False))

# Create visualization
plt.figure(figsize=(12, 6))
metrics = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*1.5, results_df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('assets/all_models_comparison.png')
print("\n✅ Comparison plot saved to assets/all_models_comparison.png")