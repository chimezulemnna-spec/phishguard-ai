# step5_baseline_model_fixed.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('src')
print("="*60)
print("Training Baseline Phishing Detection Model")
print("="*60)

# Load data
df = pd.read_csv('PILWD_features.csv')

# Prepare features and target
feature_cols = [col for col in df.columns if col != 'label']
X = df[feature_cols]
y = df['label']

print(f"\n📊 Data shape: {X.shape}")
print(f"🎯 Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Train set: {X_train.shape} ({y_train.mean()*100:.2f}% phishing)")
print(f"📈 Test set:  {X_test.shape} ({y_test.mean()*100:.2f}% phishing)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle imbalance
print("\n🔄 Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"   After SMOTE: {X_train_resampled.shape} ({y_train_resampled.mean()*100:.2f}% phishing)")

# Train multiple models for comparison
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5, 
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print('='*50)
    
    # Train
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'roc_auc': roc_auc,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    print(f"\n📊 {name} Performance:")
    print(f"  ROC-AUC Score: {roc_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:5d}  ({tn/(tn+fp)*100:.1f}% of legitimate correct)")
    print(f"  False Positives: {fp:5d}  ({fp/(tn+fp)*100:.1f}% of legitimate blocked)")
    print(f"  False Negatives: {fn:5d}  ({fn/(tp+fn)*100:.1f}% of phishing missed)")
    print(f"  True Positives:  {tp:5d}  ({tp/(tp+fn)*100:.1f}% of phishing caught)")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
best_roc_auc = results[best_model_name]['roc_auc']

print(f"\n{'='*60}")
print(f"✅ Best Model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
print('='*60)

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 20 Most Important Features:")
    print("-" * 60)
    for i, row in feature_importance.head(20).iterrows():
        if row['feature'].startswith('U'):
            category = '🌐 URL'
        elif row['feature'].startswith('H'):
            category = '📄 Content'
        else:
            category = '📊 Other'
        print(f"  {i+1:2d}. {category} | {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance_pilwd.csv', index=False)

# Group feature importance by category
print("\n📊 Feature Importance by Category:")
print("-" * 60)
url_importance = sum(results[best_model_name]['model'].feature_importances_[i] 
                     for i, f in enumerate(feature_cols) if f.startswith('U'))
content_importance = sum(results[best_model_name]['model'].feature_importances_[i] 
                        for i, f in enumerate(feature_cols) if f.startswith('H'))
other_importance = sum(results[best_model_name]['model'].feature_importances_[i] 
                      for i, f in enumerate(feature_cols) if not (f.startswith('U') or f.startswith('H')))

print(f"  🌐 URL Features:     {url_importance:.2%}")
print(f"  📄 Content Features: {content_importance:.2%}")
print(f"  📊 Other Features:   {other_importance:.2%}")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler_pilwd.pkl')
joblib.dump(smote, 'models/smote_pilwd.pkl')

# Save all models
for name, result in results.items():
    model_path = f'models/{name.lower().replace(" ", "_")}_pilwd.pkl'
    joblib.dump(result['model'], model_path)
    print(f"\n✅ Saved {name} to {model_path}")

# Save the feature mapping
feature_mapping = {
    'feature_names': feature_cols,
    'url_features': [f for f in feature_cols if f.startswith('U')],
    'content_features': [f for f in feature_cols if f.startswith('H')],
    'other_features': [f for f in feature_cols if f.startswith(('Y', 'N'))]
}
joblib.dump(feature_mapping, 'models/feature_mapping_pilwd.pkl')

print("\n✅ All models and artifacts saved successfully!")

# Create performance comparison plot
plt.figure(figsize=(10, 6))
models_list = list(results.keys())
roc_aucs = [results[m]['roc_auc'] for m in models_list]
f1_scores = [results[m]['f1'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35

plt.bar(x - width/2, roc_aucs, width, label='ROC-AUC', color='blue', alpha=0.7)
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='green', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models_list, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('assets/model_comparison_pilwd.png')
print("\n✅ Performance comparison plot saved to assets/model_comparison_pilwd.png")