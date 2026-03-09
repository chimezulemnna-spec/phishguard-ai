# step6_integrate_with_ensemble_fixed.py
import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append('src')

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("Integrating PILWD Models with Existing Ensemble")
print("="*60)

# Load the PILWD dataset for testing
df = pd.read_csv('PILWD_features.csv')
feature_cols = [col for col in df.columns if col != 'label']
X = df[feature_cols]
y = df['label']

# Load the scaler and preprocess
scaler = joblib.load('models/scaler_pilwd.pkl')
X_scaled = scaler.transform(X)

# Load the new models
new_models = {
    'RF_PILWD': joblib.load('models/random_forest_pilwd.pkl'),
    'XGB_PILWD': joblib.load('models/xgboost_pilwd.pkl'),
    'GB_PILWD': joblib.load('models/gradient_boosting_pilwd.pkl')
}

print("\n✅ Loaded new PILWD models:")
for name in new_models.keys():
    print(f"  - {name}")

# Check if existing models exist and can be loaded
existing_models = {}
existing_paths = {
    'RF_Existing': 'models/rf_model.pkl',
    'XGB_Existing': 'models/xgb_model.pkl',
    'Voting_Existing': 'models/voting_model.pkl'
}

for name, path in existing_paths.items():
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            existing_models[name] = model
            print(f"  - {name} (loaded)")
        except Exception as e:
            print(f"  - {name} (found but couldn't load: {e})")

print(f"\n📊 Testing on {len(X)} samples...")

# Get predictions from all models
predictions = {}
probabilities = {}

# New models predictions
for name, model in new_models.items():
    probabilities[name] = model.predict_proba(X_scaled)[:, 1]
    predictions[name] = (probabilities[name] > 0.5).astype(int)
    roc_auc = roc_auc_score(y, probabilities[name])
    print(f"\n{name} ROC-AUC: {roc_auc:.4f}")

# Create ensemble of new models
print("\n" + "="*60)
print("Creating Ensemble of PILWD Models")
print("="*60)

# Fix: Take a stratified sample to ensure both classes are present
from sklearn.model_selection import train_test_split

# Take a smaller stratified sample for fitting the ensemble
X_sample, _, y_sample, _ = train_test_split(
    X_scaled, y, train_size=2000, random_state=42, stratify=y
)

print(f"Training ensemble on {len(X_sample)} samples with class distribution:")
print(y_sample.value_counts())

# Simple voting ensemble
voting_ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in new_models.items()],
    voting='soft'
)
voting_ensemble.fit(X_sample, y_sample)  # Fit on stratified sample
ensemble_proba = voting_ensemble.predict_proba(X_scaled)[:, 1]
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print(f"\n📊 PILWD Ensemble Performance:")
print(f"  ROC-AUC: {roc_auc_score(y, ensemble_proba):.4f}")
print(f"\nClassification Report:")
print(classification_report(y, ensemble_pred, target_names=['Legitimate', 'Phishing']))

# Create weighted ensemble based on individual performance
print("\n" + "="*60)
print("Creating Weighted Ensemble")
print("="*60)

# Calculate weights based on ROC-AUC scores
weights = {
    'RF_PILWD': 0.9974,
    'XGB_PILWD': 0.9955,
    'GB_PILWD': 0.9944
}

# Normalize weights
total = sum(weights.values())
for model in weights:
    weights[model] /= total

print("\n⚖️ Model Weights:")
for model, weight in weights.items():
    print(f"  {model}: {weight:.3f}")

# Calculate weighted probability
weighted_proba = np.zeros(len(y))
for name, model in new_models.items():
    weighted_proba += weights[name] * probabilities[name]

weighted_pred = (weighted_proba > 0.5).astype(int)

print(f"\n📊 Weighted Ensemble Performance:")
print(f"  ROC-AUC: {roc_auc_score(y, weighted_proba):.4f}")
cm = confusion_matrix(y, weighted_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:5d} ({tn/(tn+fp)*100:.1f}%)")
print(f"  False Positives: {fp:5d} ({fp/(tn+fp)*100:.1f}%)")
print(f"  False Negatives: {fn:5d} ({fn/(tp+fn)*100:.1f}%)")
print(f"  True Positives:  {tp:5d} ({tp/(tp+fn)*100:.1f}%)")

# Save the ensemble models
print("\n" + "="*60)
print("Saving Ensemble Models")
print("="*60)

os.makedirs('models', exist_ok=True)

# Save voting ensemble
joblib.dump(voting_ensemble, 'models/voting_ensemble_pilwd.pkl')
print("✅ Saved voting_ensemble_pilwd.pkl")

# Save weighted ensemble info
ensemble_info = {
    'models': new_models,
    'weights': weights,
    'feature_names': feature_cols,
    'scaler': scaler
}
joblib.dump(ensemble_info, 'models/weighted_ensemble_pilwd.pkl')
print("✅ Saved weighted_ensemble_pilwd.pkl")

# Create a combined predictor that can use both old and new models
print("\n" + "="*60)
print("Creating Unified Predictor")
print("="*60)

class UnifiedPhishingPredictor:
    """
    Unified predictor that can use both existing and PILWD-trained models
    """
    def __init__(self, use_pilwd=True, use_existing=False):
        self.use_pilwd = use_pilwd
        self.use_existing = use_existing
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.weights = None
        
        # Load PILWD models if requested
        if use_pilwd:
            try:
                ensemble_info = joblib.load('models/weighted_ensemble_pilwd.pkl')
                self.models = ensemble_info['models']
                self.weights = ensemble_info['weights']
                self.scaler = ensemble_info['scaler']
                self.feature_names = ensemble_info['feature_names']
                print("✅ Loaded PILWD models")
            except Exception as e:
                print(f"⚠️ Could not load PILWD models: {e}")
        
        # Load existing models if requested
        if use_existing:
            # This will need your existing feature extractors
            print("ℹ️ Existing model integration requires feature extraction")
    
    def predict_from_features(self, features):
        """
        Predict using pre-computed features
        features: array of 53 features in correct order
        """
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get weighted probability
        proba = 0
        for name, model in self.models.items():
            proba += self.weights[name] * model.predict_proba(features_scaled)[0, 1]
        
        pred = 1 if proba > 0.5 else 0
        
        return {
            'probability': float(proba),
            'prediction': 'phishing' if pred == 1 else 'legitimate',
            'confidence': max(proba, 1-proba),
            'model_used': 'PILWD Ensemble'
        }
    
    def get_feature_importance(self):
        """Get feature importance from the Random Forest model"""
        if 'RF_PILWD' in self.models:
            rf_model = self.models['RF_PILWD']
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None
    
    def batch_predict(self, features_list):
        """Predict multiple samples at once"""
        features_array = np.array(features_list)
        features_scaled = self.scaler.transform(features_array)
        
        # Get weighted probabilities for all samples
        probas = np.zeros(len(features_array))
        for name, model in self.models.items():
            probas += self.weights[name] * model.predict_proba(features_scaled)[:, 1]
        
        predictions = (probas > 0.5).astype(int)
        
        return [{
            'probability': float(proba),
            'prediction': 'phishing' if pred == 1 else 'legitimate',
            'confidence': max(proba, 1-proba)
        } for proba, pred in zip(probas, predictions)]

# Test the unified predictor
predictor = UnifiedPhishingPredictor(use_pilwd=True, use_existing=False)

# Test on a few samples
print("\n" + "="*60)
print("Testing Unified Predictor")
print("="*60)

# Take stratified samples for testing
test_samples = pd.concat([
    df[df['label']==0].sample(3, random_state=42),
    df[df['label']==1].sample(3, random_state=42)
])

for i, (idx, row) in enumerate(test_samples.iterrows()):
    features = row[feature_cols].values
    true_label = row['label']
    
    result = predictor.predict_from_features(features)
    
    status = "✅ CORRECT" if (true_label == 1 and result['prediction'] == 'phishing') or \
                            (true_label == 0 and result['prediction'] == 'legitimate') else "❌ WRONG"
    
    print(f"\nTest Sample {i+1} {status}:")
    print(f"  True: {'Phishing' if true_label == 1 else 'Legitimate'}")
    print(f"  Predicted: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probability: {result['probability']:.4f}")

print("\n✅ Integration complete!")