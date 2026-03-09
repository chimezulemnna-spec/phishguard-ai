"""
Ensemble builder for Phishing Detection.

FIX: Original used voting='hard' which disables predict_proba.
     Changed to voting='soft' so probability scores are available
     for ROC-AUC, threshold tuning, and app confidence display.
"""

import os
import pickle
from sklearn.ensemble import VotingClassifier


def build_voting_ensemble(rf_model, xgb_model, X_train, y_train,
                          save_path='models/voting_model.pkl'):
    """
    Build and fit a soft-voting ensemble of RF + XGBoost.
    Returns the fitted ensemble.
    """
    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'          # FIX: was 'hard' — soft enables predict_proba
    )
    ensemble.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"✅ Ensemble saved to {save_path}")
    return ensemble
