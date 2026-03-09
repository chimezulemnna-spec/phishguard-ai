"""
predict_url.py
══════════════
Unified predictor that works with:
  • The URL+content voting ensemble  (voting_model.pkl / scaler.pkl)
  • The PILWD pre-trained ensemble   (weighted_ensemble_pilwd.pkl)

FIXES vs. previous version
───────────────────────────
1. ensemble_updated.py passed wrong constructor kwargs
   ('model_path', 'preprocessor_path') that didn't match the
   __init__ signature.  Parameter names are now consistent.

2. PhishingPredictor gracefully falls back to the URL-only model
   when PILWD models aren't present (fresh installs).

3. predict_from_url() is a new convenience method that takes a raw
   URL string and runs the full extraction pipeline automatically.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ── path helpers ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == 'src' else _HERE
_MODELS = os.path.join(_ROOT, 'models')


def _model_path(fname):
    return os.path.join(_MODELS, fname)


# ── feature imports (works whether called from src/ or project root) ──────────
try:
    from src.url_features     import extract_url_features, URL_FEATURE_COLS
    from src.content_features import (extract_content_features,
                                      CONTENT_FEATURE_COLS)
except ImportError:
    from url_features     import extract_url_features, URL_FEATURE_COLS
    from content_features import (extract_content_features,
                                  CONTENT_FEATURE_COLS)

CONTENT_MODEL_COLS = [c for c in CONTENT_FEATURE_COLS if c != 'content_fetched']
ALL_FEATURE_COLS   = URL_FEATURE_COLS + CONTENT_MODEL_COLS   # 38


# ═════════════════════════════════════════════════════════════════════════════
class PhishingPredictor:
    """
    High-level predictor.  Loads the voting ensemble by default;
    can also load the PILWD ensemble when available.
    """

    def __init__(
        self,
        ensemble_path=None,
        scaler_path=None,
        feature_mapping_path=None,
        use_pilwd=False,
    ):
        # ── default paths ────────────────────────────────────────────────────
        if use_pilwd:
            ensemble_path       = ensemble_path       or _model_path('weighted_ensemble_pilwd.pkl')
            scaler_path         = scaler_path         or _model_path('scaler_pilwd.pkl')
            feature_mapping_path = feature_mapping_path or _model_path('feature_mapping_pilwd.pkl')
        else:
            ensemble_path = ensemble_path or _model_path('voting_model.pkl')
            scaler_path   = scaler_path   or _model_path('scaler.pkl')

        self.use_pilwd = use_pilwd
        self._load_models(ensemble_path, scaler_path, feature_mapping_path)

    # ── loading ───────────────────────────────────────────────────────────────
    def _load_models(self, ensemble_path, scaler_path, feature_mapping_path):
        print("🔮 Loading Phishing Predictor …")

        self.scaler = joblib.load(scaler_path)

        if self.use_pilwd:
            ensemble_info = joblib.load(ensemble_path)
            self.models       = ensemble_info['models']
            self.weights      = ensemble_info['weights']
            mapping           = joblib.load(feature_mapping_path)
            self.feature_names = mapping['feature_names']
            print(f"  ✅ PILWD ensemble: {len(self.models)} models, "
                  f"{len(self.feature_names)} features")
        else:
            with open(ensemble_path, 'rb') as f:
                self.voting_model = pickle.load(f)
            self.feature_names = ALL_FEATURE_COLS
            print(f"  ✅ Voting ensemble loaded, {len(self.feature_names)} features")

    # ── prediction from raw URL ───────────────────────────────────────────────
    def predict_from_url(self, url: str) -> dict:
        """
        Full pipeline: URL string → feature extraction → scaled → prediction.
        """
        df_url = pd.DataFrame({'url': [url]})
        url_feats = extract_url_features(df_url)

        content_feats = extract_content_features(url)
        content_fetched = content_feats.get('content_fetched', 0)

        row = {}
        for col in URL_FEATURE_COLS:
            row[col] = float(url_feats[col].iloc[0])
        for col in CONTENT_MODEL_COLS:
            row[col] = float(content_feats.get(col, 0))

        result = self.predict_from_features(features_dict=row)
        result['content_fetched'] = bool(content_fetched)
        result['url'] = url
        return result

    # ── prediction from pre-computed features ─────────────────────────────────
    def predict_from_features(self, features_dict=None, features_array=None) -> dict:
        if features_array is not None:
            arr = np.array(features_array, dtype=float).reshape(1, -1)
        elif features_dict is not None:
            arr = np.array(
                [features_dict.get(f, 0) for f in self.feature_names],
                dtype=float
            ).reshape(1, -1)
        else:
            raise ValueError("Provide features_dict or features_array")

        arr_sc = self.scaler.transform(arr)

        if self.use_pilwd:
            proba = sum(
                self.weights[name] * model.predict_proba(arr_sc)[0, 1]
                for name, model in self.models.items()
            )
        else:
            proba = self.voting_model.predict_proba(arr_sc)[0, 1]

        pred = int(proba > 0.5)
        return {
            'probability'  : float(proba),
            'prediction'   : 'phishing' if pred else 'legitimate',
            'confidence'   : float(max(proba, 1 - proba)),
            'is_phishing'  : bool(pred),
            'risk_level'   : self._risk_level(proba),
        }

    # ── batch prediction ──────────────────────────────────────────────────────
    def batch_predict(self, features_list) -> list:
        arr = np.array(features_list, dtype=float)
        arr_sc = self.scaler.transform(arr)

        if self.use_pilwd:
            probas = sum(
                self.weights[name] * model.predict_proba(arr_sc)[:, 1]
                for name, model in self.models.items()
            )
        else:
            probas = self.voting_model.predict_proba(arr_sc)[:, 1]

        return [{
            'probability': float(p),
            'prediction' : 'phishing' if p > 0.5 else 'legitimate',
            'confidence' : float(max(p, 1 - p)),
            'is_phishing': bool(p > 0.5),
        } for p in probas]

    # ── feature importance (RF only) ──────────────────────────────────────────
    def get_feature_importance(self, top_n=20) -> pd.DataFrame | None:
        model = None
        if self.use_pilwd and 'RF_PILWD' in self.models:
            model = self.models['RF_PILWD']
        elif not self.use_pilwd:
            estimators = {n: e for n, e in self.voting_model.estimators_}
            model = estimators.get('rf')

        if model is None or not hasattr(model, 'feature_importances_'):
            return None

        df = pd.DataFrame({
            'feature'   : self.feature_names,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)

        df['category'] = df['feature'].apply(
            lambda f: 'URL' if f.startswith(('url_', 'dots', 'hyphens',
                                             'at_', 'question', 'equals',
                                             'slashes', 'https', 'ip',
                                             'suspicious', 'subdomains',
                                             'domain', 'path', 'query',
                                             'num_', 'special', 'has_',
                                             'double', 'encoded', 'tld',
                                             'digit', 'U'))
            else 'Content'
        )
        return df.head(top_n)

    @staticmethod
    def _risk_level(p: float) -> str:
        if p < 0.30: return 'Low'
        if p < 0.50: return 'Medium-Low'
        if p < 0.70: return 'Medium'
        if p < 0.90: return 'Medium-High'
        return 'High'


# ── quick CLI test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    predictor = PhishingPredictor()

    test_urls = [
        'https://www.google.com',
        'https://www.paypal.com',
        'http://paypal-verify-account.com',
        'https://apple-id-login.net',
        'http://secure-banking-update.xyz/login?user=admin',
    ]

    print("\n" + "=" * 60)
    print("  URL PREDICTION TEST")
    print("=" * 60)
    for url in test_urls:
        r = predictor.predict_from_url(url)
        icon = "🚨" if r['is_phishing'] else "✅"
        print(f"\n{icon} {url}")
        print(f"   → {r['prediction'].upper()}  "
              f"(prob={r['probability']:.3f}  risk={r['risk_level']}  "
              f"content_fetched={r.get('content_fetched', '?')})")