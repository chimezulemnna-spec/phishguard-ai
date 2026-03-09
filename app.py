"""
PhishGuard AI  –  Streamlit App
================================
Run:  streamlit run app.py

FIXES vs previous version
──────────────────────────
1. st.write() debug lines removed from analyze_url() (leaked into UI).
2. predict_with_original() mutated the shared feature list with .remove();
   now always works on a local copy.
3. content_available flag now reads content_fetched key correctly.
4. PILWD ensemble fully wired up — no longer falls back to original model.
5. _predict_pilwd() handles missing PILWD features gracefully (fills 0).
6. Spline viewer height corrected (container + iframe both 400 px).
7. All HTML blocks have matching open/close tags.
8. Scroll-progress <script> had mismatched braces; rewritten cleanly.
9. Model status messages moved out of load_models() so sidebar widgets
   never create duplicate-key errors on re-runs.
"""

import os
import sys
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.url_features     import extract_url_features, URL_FEATURE_COLS
from src.content_features import (extract_content_features,
                                  content_feature_names, CONTENT_FEATURE_COLS)

# ── constants ──────────────────────────────────────────────────────────────────
CONTENT_MODEL_COLS = [c for c in CONTENT_FEATURE_COLS if c != "content_fetched"]
ALL_FEATURE_COLS   = URL_FEATURE_COLS + CONTENT_MODEL_COLS   # 38

SPLINE_URLS = {
    "neutral":  "https://prod.spline.design/EKxqFdoVg9IgieUr/scene.splinecode",
    "safe":     "https://prod.spline.design/BgtiSLgZ47d5N9Ik/scene.splinecode",
    "phishing": "https://prod.spline.design/OHp2a9nULAIVQF0Z/scene.splinecode",
}

# ── page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="PhishGuard – Advanced Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
def _load_css() -> str:
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            base = f.read()
    else:
        base = ""   # styles.css missing — app will still work, just unstyled

    addendum = """
    /* ── hero text (not in styles.css) ── */
    .hero-title {
        font-size: 3rem; font-weight: 800; line-height: 1.1; margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 2rem;
    }

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }

    /* ── toast notifications ── */
    .pg-toast {
        position: fixed;
        bottom: 32px;
        right: 32px;
        z-index: 9999;
        padding: 14px 22px;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 600;
        color: #fff;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.35s ease, transform 0.35s ease;
        pointer-events: none;
    }
    .pg-toast.show {
        opacity: 1;
        transform: translateY(0);
    }
    .pg-toast.safe     { background: linear-gradient(135deg, #00C853, #007A33); border-left: 4px solid #00FF6A; }
    .pg-toast.phishing { background: linear-gradient(135deg, #FF3D00, #8B0000); border-left: 4px solid #FF6D00; }

    /* ── streamlit overrides ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header   { visibility: hidden; }
    .block-container            { padding-top: 0 !important; max-width: 1200px; }
    .stApp                      { background: var(--bg-primary); color: var(--text-primary); }
    """
    return base + addendum

st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  3-D Spline shield
# ══════════════════════════════════════════════════════════════════════════════
def show_shield(state: str = "neutral"):
    url  = SPLINE_URLS[state]
    html = f"""
<div style="width:100%;height:400px;border-radius:16px;overflow:hidden;">
  <script type="module"
    src="https://unpkg.com/@splinetool/viewer@1.9.5/build/spline-viewer.js">
  </script>
  <spline-viewer url="{url}" style="width:100%;height:100%;"></spline-viewer>
</div>"""
    components.html(html, height=400)


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading detection models…")
def load_models() -> dict:
    m: dict = {}

    # ── original 38-feature voting ensemble ───────────────────────────────────
    try:
        m["original_voting"] = pickle.load(open("models/voting_model.pkl", "rb"))
        m["original_scaler"] = pickle.load(open("models/scaler.pkl",       "rb"))
        m["_orig_ok"]  = True
    except Exception as e:
        m["_orig_ok"]  = False
        m["_orig_err"] = str(e)

    # ── PILWD 53-feature weighted ensemble ────────────────────────────────────
    try:
        ei = joblib.load("models/weighted_ensemble_pilwd.pkl")
        m["pilwd_models"]    = ei["models"]
        m["pilwd_weights"]   = ei["weights"]
        m["pilwd_scaler"]    = ei["scaler"]
        m["pilwd_feat_names"]= ei["feature_names"]
        m["_pilwd_ok"]  = True
    except Exception as e:
        m["_pilwd_ok"]  = False
        m["_pilwd_err"] = str(e)

    return m


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_all_features(url: str):
    """Returns (feature_df, content_fetched_bool)."""
    df  = extract_url_features(pd.DataFrame({"url": [url]}))
    raw = extract_content_features(url)
    for col in CONTENT_FEATURE_COLS:
        df[col] = raw.get(col, 0)
    return df, bool(raw.get("content_fetched", 0))


# ══════════════════════════════════════════════════════════════════════════════
#  WHOIS domain age
# ══════════════════════════════════════════════════════════════════════════════
def get_domain_age(url: str) -> dict:
    """Returns dict with age_days, creation_date, registrar, age_str."""
    try:
        import whois
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list): cd = cd[0]
        if cd is None:
            return {"age_days": None, "creation_date": "Unknown",
                    "registrar": str(w.registrar or "Unknown"), "age_str": "Unknown"}
        from datetime import datetime, timezone
        now  = datetime.now(timezone.utc)
        if cd.tzinfo is None: cd = cd.replace(tzinfo=timezone.utc)
        age_days = (now - cd).days
        years    = age_days // 365
        months   = (age_days % 365) // 30
        if years > 0:
            age_str = f"{years}y {months}m"
        elif months > 0:
            age_str = f"{months} months"
        else:
            age_str = f"{age_days} days"
        return {
            "age_days":      age_days,
            "creation_date": cd.strftime("%b %d, %Y"),
            "registrar":     str(w.registrar or "Unknown")[:40],
            "age_str":       age_str,
        }
    except Exception:
        return {"age_days": None, "creation_date": "Unknown",
                "registrar": "Unknown", "age_str": "Unavailable"}


# ══════════════════════════════════════════════════════════════════════════════
#  SSL certificate check
# ══════════════════════════════════════════════════════════════════════════════
def get_ssl_info(url: str) -> dict:
    """Returns dict with valid, issuer, expires, days_left, error."""
    try:
        import ssl, socket
        from urllib.parse import urlparse
        from datetime import datetime, timezone

        hostname = urlparse(url).netloc.replace("www.", "").split(":")[0]
        ctx      = ssl.create_default_context()
        with ctx.wrap_socket(socket.create_connection((hostname, 443), timeout=5),
                             server_hostname=hostname) as s:
            cert     = s.getpeercert()

        # Expiry
        exp_str  = cert.get("notAfter", "")
        exp_dt   = datetime.strptime(exp_str, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        days_left= (exp_dt - datetime.now(timezone.utc)).days

        # Issuer
        issuer_dict = dict(x[0] for x in cert.get("issuer", []))
        issuer  = issuer_dict.get("organizationName", issuer_dict.get("commonName", "Unknown"))[:35]

        return {
            "valid":     True,
            "issuer":    issuer,
            "expires":   exp_dt.strftime("%b %d, %Y"),
            "days_left": days_left,
            "error":     None,
        }
    except ssl.SSLCertVerificationError:
        return {"valid": False, "issuer": "Invalid / Untrusted",
                "expires": "—", "days_left": None, "error": "Certificate not trusted"}
    except Exception as e:
        return {"valid": None, "issuer": "Unavailable",
                "expires": "—", "days_left": None, "error": str(e)[:60]}


# ══════════════════════════════════════════════════════════════════════════════
#  Google Safe Browsing API
# ══════════════════════════════════════════════════════════════════════════════
def check_safe_browsing(url: str) -> dict:
    """Check URL against Google Safe Browsing API."""
    try:
        import urllib.request
        API_KEY  = (st.secrets.get("GOOGLE_SAFE_BROWSING_KEY")
                     or "AIzaSyBXbnwQTWXslyqM9uKUdbbiX36wLcK_4jw")  # fallback for local dev
        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={API_KEY}"
        payload  = json.dumps({
            "client": {"clientId": "phishguard", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes":      ["MALWARE","SOCIAL_ENGINEERING","UNWANTED_SOFTWARE","POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes":    ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries":    [{"url": url}],
            }
        }).encode()
        req  = urllib.request.Request(endpoint, data=payload,
                                      headers={"Content-Type":"application/json"})
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        if data.get("matches"):
            threat = data["matches"][0].get("threatType","UNKNOWN")
            return {"flagged": True,  "threat": threat.replace("_"," ").title(),
                    "error": None}
        return {"flagged": False, "threat": None, "error": None}
    except Exception as e:
        return {"flagged": None, "threat": None, "error": str(e)[:60]}


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction
# ══════════════════════════════════════════════════════════════════════════════
def _predict_original(models: dict, df: pd.DataFrame):
    cols = list(ALL_FEATURE_COLS)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    X_sc  = models["original_scaler"].transform(df[cols].values)
    pred  = models["original_voting"].predict(X_sc)[0]
    proba = models["original_voting"].predict_proba(X_sc)[0]
    return int(pred), proba


# PILWD coded feature name -> our extracted feature column
PILWD_FEATURE_MAP = {
    "U1":"url_length",       "U2":"dots",            "U3":"hyphens",
    "U5.1":"question_marks", "U5.2":"equals",        "U5.3":"slashes",
    "U6.1":"https",          "U6.2":"ip",            "U6.3":"has_port",
    "U7.1":"subdomains",     "U7.2":"domain_length", "U7.3":"path_length",
    "U7.4":"query_length",   "U7.5":"tld_length",    "U7.6":"url_entropy",
    "U7.7":"double_slash",   "U7.8":"encoded_chars", "U7.9":"at_symbol",
    "U8.2":"https_count",    "U_Sensitive_Count":"suspicious_words",
    "U_Special_Ratio":"digit_letter_ratio",
    "H1.1":"content_num_forms",            "H1.2":"content_has_password_field",
    "H1.3":"content_suspicious_form_action","H1.4":"content_has_title",
    "H1.5":"content_title_has_brand",      "H_Ext_Ratio":"content_external_link_ratio",
    "H_Null_Ratio":"content_external_favicon","H_Script_Count":"content_has_js_redirect",
    "H_Iframe_Count":"content_num_iframes","H_Hidden_Count":"content_num_hidden_elements",
    "H_Form_Count":"content_brand_keyword_count",
    "H_Suspicious_Form":"content_suspicious_phrase_count",
    "NU4":"num_digits",      "NU8.1":"num_letters",  "NH2.1":"special_chars",
    "NH2.2":"num_parameters","NH3":"encoded_chars",
    "Y1":"https",            "NY2":"content_has_password_field",
    "NY3.1":"subdomains",    "NY3.2":"domain_length","NY3.3":"tld_length",
    "NY4.1":"url_entropy",   "NY4.2":"special_chars","NY4.3":"encoded_chars",
    "NY5.1":"content_external_link_ratio", "NY5.2":"content_num_iframes",
    "NY6":"content_has_js_redirect",       "Y7":"suspicious_words",
    "Y8":"content_suspicious_phrase_count","NY9":"content_has_alert",
    "NLP_Urgent_Count":"content_suspicious_phrase_count",
}


def _predict_pilwd(models: dict, df: pd.DataFrame):
    feat_names = models["pilwd_feat_names"]
    # Map PILWD coded names to our extracted feature values
    row = []
    for f in feat_names:
        our_col = PILWD_FEATURE_MAP.get(f)          # look up mapping
        if our_col and our_col in df.columns:
            row.append(float(df[our_col].iloc[0]))   # use mapped value
        elif f in df.columns:
            row.append(float(df[f].iloc[0]))          # fallback: direct match
        else:
            row.append(0.0)                           # unknown feature
    X = np.array([row])
    X_sc = models["pilwd_scaler"].transform(X)
    p    = sum(
        models["pilwd_weights"][name] * model.predict_proba(X_sc)[0, 1]
        for name, model in models["pilwd_models"].items()
    )
    return int(p > 0.5), np.array([1 - p, p])


def analyze_url(url: str, models: dict, use_pilwd: bool = False) -> dict:
    df, content_fetched = extract_all_features(url)

    if use_pilwd and models.get("_pilwd_ok"):
        pred, proba = _predict_pilwd(models, df)
        label = "PILWD Weighted Ensemble (RF + XGB + GB)"
    elif models.get("_orig_ok"):
        pred, proba = _predict_original(models, df)
        label = "Original Voting Ensemble (RF + XGB)"
    else:
        raise RuntimeError("No models loaded. Check models/ folder.")

    whois_info = get_domain_age(url)
    ssl_info   = get_ssl_info(url)
    gsb_info   = check_safe_browsing(url)
    return {
        "prediction":      pred,
        "confidence":      float(max(proba)) * 100,
        "phishing_prob":   float(proba[1])   * 100,
        "legit_prob":      float(proba[0])   * 100,
        "features":        df.iloc[0].to_dict(),
        "content_fetched": content_fetched,
        "model_used":      label,
        "whois":           whois_info,
        "ssl":             ssl_info,
        "gsb":             gsb_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Load models + sidebar
# ══════════════════════════════════════════════════════════════════════════════
models = load_models()

# ── Scan history (session state) ──────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

with st.sidebar:
    st.markdown('''<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 115" width="36" height="36">
  <defs>
    <linearGradient id="sg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#4A9EFF"/>
      <stop offset="100%" stop-color="#0066CC"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- Shield body -->
  <path d="M50 4 L92 20 L92 55 C92 80 72 100 50 111 C28 100 8 80 8 55 L8 20 Z"
        fill="url(#sg)" filter="url(#glow)" opacity="0.95"/>
  <!-- Inner shield border -->
  <path d="M50 12 L84 26 L84 55 C84 76 67 94 50 104 C33 94 16 76 16 55 L16 26 Z"
        fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="1.5"/>
  <!-- Check mark -->
  <polyline points="32,57 44,70 68,42"
            fill="none" stroke="#ffffff" stroke-width="7"
            stroke-linecap="round" stroke-linejoin="round"
            filter="url(#glow)"/>
</svg>
  <span style="font-size:22px;font-weight:700;color:var(--text-primary);">PhishGuard AI</span>
</div>''', unsafe_allow_html=True)
    st.subheader("Model Configuration")

    model_option = st.radio(
        "Detection Model:",
        ["Original Ensemble (URL + Content)", "PILWD Ensemble (99.65% ROC-AUC) – Beta"],
        index=0,
    )
    use_pilwd = "PILWD" in model_option

    st.subheader("System Status")
    if models.get("_orig_ok"):
        st.success("✅ Original Models: Ready")
    else:
        st.error(f"❌ Original: {models.get('_orig_err', 'not found')}")

    if models.get("_pilwd_ok"):
        st.success("✅ PILWD Models: Ready")
        st.markdown("##### Model Weights")
        for name, w in models["pilwd_weights"].items():
            st.metric(label=name, value=f"{w:.1%}")
    else:
        st.warning(f"⚠️ PILWD: {models.get('_pilwd_err', 'not found')}")


# ══════════════════════════════════════════════════════════════════════════════
#  Page layout
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="progress-container"><div class="progress-bar" id="pgbar"></div></div>',
            unsafe_allow_html=True)

st.markdown("""
<style>
.navbar-item {
    color: #ffffff !important;
    text-decoration: none !important;
    font-size: 15px;
    font-weight: 500;
    position: relative;
    display: inline-block;
    cursor: pointer;
    padding-bottom: 4px;
}
.navbar-item::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 0; height: 2px;
    background: #4A9EFF;
    box-shadow: 0 0 8px #4A9EFF, 0 0 16px #4A9EFF;
    border-radius: 2px;
    transition: width 0.3s ease;
}
.navbar-item:hover { color: #4A9EFF !important; }
.navbar-item:hover::after { width: 100% !important; }

/* ── Hamburger ── */
.pg-hamburger {
    display: none;
    flex-direction: column;
    justify-content: space-between;
    width: 26px; height: 18px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    z-index: 1100;
}
.pg-hamburger span {
    display: block;
    width: 100%; height: 2px;
    background: #ffffff;
    border-radius: 2px;
    transition: all 0.3s ease;
}
.pg-hamburger.is-open span:nth-child(1) { transform: translateY(8px) rotate(45deg); }
.pg-hamburger.is-open span:nth-child(2) { opacity: 0; transform: scaleX(0); }
.pg-hamburger.is-open span:nth-child(3) { transform: translateY(-8px) rotate(-45deg); }

@media (max-width: 768px) {
    .pg-hamburger { display: flex !important; }
    .navbar-menu  { display: none !important; }
    .navbar-menu.is-open {
        display: flex !important;
        position: fixed !important;
        top: 65px !important; left: 0 !important;
        width: 100% !important;
        background: rgba(10,15,26,0.97) !important;
        backdrop-filter: blur(14px) !important;
        flex-direction: column !important;
        align-items: center !important;
        padding: 12px 0 20px !important;
        margin: 0 !important;
        list-style: none !important;
        z-index: 1050 !important;
        border-bottom: 1px solid rgba(74,158,255,0.25) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
        animation: pgSlideDown 0.25s ease forwards;
    }
    .navbar-menu.is-open li {
        width: 100%;
        text-align: center;
        padding: 14px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .navbar-menu.is-open .navbar-item {
        font-size: 16px !important;
        letter-spacing: 0.5px;
    }
    @keyframes pgSlideDown {
        from { opacity:0; transform: translateY(-10px); }
        to   { opacity:1; transform: translateY(0); }
    }
}
</style>
<div class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 115" width="36" height="36">
  <defs>
    <linearGradient id="sg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#4A9EFF"/>
      <stop offset="100%" stop-color="#0066CC"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <path d="M50 4 L92 20 L92 55 C92 80 72 100 50 111 C28 100 8 80 8 55 L8 20 Z"
        fill="url(#sg)" filter="url(#glow)" opacity="0.95"/>
  <path d="M50 12 L84 26 L84 55 C84 76 67 94 50 104 C33 94 16 76 16 55 L16 26 Z"
        fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="1.5"/>
  <polyline points="32,57 44,70 68,42"
            fill="none" stroke="#ffffff" stroke-width="7"
            stroke-linecap="round" stroke-linejoin="round"
            filter="url(#glow)"/>
</svg></div>
    <span class="navbar-title">PhishGuard AI</span>
  </div>
  <ul class="navbar-menu" id="pg-nav-menu">
    <li><a class="navbar-item" data-target="home">Home</a></li>
    <li><a class="navbar-item" data-target="how-it-works">How It Works</a></li>
    <li><a class="navbar-item" data-target="performance">Performance</a></li>
    <li><a class="navbar-item" data-target="evaluation">Evaluation</a></li>
    <li><a class="navbar-item" data-target="about">About</a></li>
  </ul>
  <button class="pg-hamburger" id="pg-hamburger" aria-label="Menu">
    <span></span><span></span><span></span>
  </button>
</div>

""", unsafe_allow_html=True)

components.html("""
<script>
// Inject scroll script directly into parent document head
// so it runs in the parent window context — no iframe boundary issues
(function() {
    var pdoc = window.parent.document;

    // Remove any previous injected script
    var old = pdoc.getElementById('_pg_scroll_script');
    if (old) old.remove();

    var s = pdoc.createElement('script');
    s.id = '_pg_scroll_script';
    s.textContent = `
        (function() {
            var OFFSET = 85;

            // ── Hamburger menu ──
            (function() {
              function initHamburger() {
                var btn  = document.getElementById('pg-hamburger');
                var menu = document.getElementById('pg-nav-menu');
                if (!btn || !menu) { setTimeout(initHamburger, 300); return; }
                btn.addEventListener('click', function() {
                  btn.classList.toggle('is-open');
                  menu.classList.toggle('is-open');
                });
                menu.querySelectorAll('.navbar-item').forEach(function(link) {
                  link.addEventListener('click', function() {
                    btn.classList.remove('is-open');
                    menu.classList.remove('is-open');
                  });
                });
              }
              initHamburger();
            })();

            function pgScroll(id) {
                var target = document.getElementById(id);
                if (!target) { console.log('[PG] target not found:', id); return; }

                // Try every possible scroll container
                var scrolled = false;
                var el = target.parentElement;
                while (el) {
                    var st = window.getComputedStyle(el);
                    var ov = st.overflow + st.overflowY;
                    if (/auto|scroll/.test(ov) && el.scrollHeight > el.clientHeight + 5) {
                        var dest = el.scrollTop + target.getBoundingClientRect().top
                                   - el.getBoundingClientRect().top - OFFSET;
                        el.scrollTo({ top: dest, behavior: 'smooth' });
                        console.log('[PG] scrolled container:', el.tagName, el.className.slice(0,40));
                        scrolled = true;
                        break;
                    }
                    el = el.parentElement;
                }

                // Always also try window
                var wDest = window.pageYOffset + target.getBoundingClientRect().top - OFFSET;
                window.scrollTo({ top: wDest, behavior: 'smooth' });
                console.log('[PG] window.scrollTo:', wDest, '| container scrolled:', scrolled);
            }

            // Remove old listener, add new
            if (window._pgClickHandler) {
                document.removeEventListener('click', window._pgClickHandler, true);
            }
            window._pgClickHandler = function(e) {
                var el = e.target;
                while (el && el.tagName) {
                    if (el.classList && el.classList.contains('navbar-item')) {
                        var t = el.getAttribute('data-target');
                        console.log('[PG] navbar click detected, target:', t);
                        if (t) {
                            e.preventDefault();
                            e.stopImmediatePropagation();
                            pgScroll(t);
                        }
                        return;
                    }
                    el = el.parentElement;
                }
            };
            document.addEventListener('click', window._pgClickHandler, true);
            console.log('[PG] scroll engine ready');

            // ── Scroll spy ──────────────────────────────────────────────
            var SECTIONS = ['home', 'how-it-works', 'performance', 'evaluation', 'about'];

            function getSpyScroller() {
                var el = document.getElementById('home');
                while (el) {
                    var st = window.getComputedStyle(el);
                    var ov = st.overflow + st.overflowY;
                    if (/auto|scroll/.test(ov) && el.scrollHeight > el.clientHeight + 5) return el;
                    el = el.parentElement;
                }
                return window;
            }

            function updateActive() {
                var scroller  = getSpyScroller();
                var scrollTop = scroller === window ? window.pageYOffset : scroller.scrollTop;
                var offset    = 140;
                var current   = SECTIONS[0];
                SECTIONS.forEach(function(id) {
                    var anchor = document.getElementById(id);
                    if (!anchor) return;
                    var rect = anchor.getBoundingClientRect();
                    var sTop = scroller === window ? 0 : scroller.getBoundingClientRect().top;
                    if (scrollTop + (rect.top - sTop) - scrollTop <= offset) {
                        // simpler: use getBoundingClientRect relative to viewport
                    }
                    // Use offsetTop walk-up for accuracy
                    var top = 0;
                    var node = anchor;
                    while (node && node !== (scroller === window ? document.body : scroller)) {
                        top += node.offsetTop;
                        node = node.offsetParent;
                    }
                    if (scrollTop >= top - offset) current = id;
                });
                document.querySelectorAll('.navbar-item').forEach(function(item) {
                    item.classList.toggle('active', item.getAttribute('data-target') === current);
                });
            }

            function attachSpy() {
                var scroller = getSpyScroller();
                if (scroller._pgSpy) return;
                scroller._pgSpy = true;
                scroller.addEventListener('scroll', updateActive);
                updateActive();
            }

            setTimeout(attachSpy, 800);
            setTimeout(attachSpy, 2500);
        })();
    `;
    pdoc.head.appendChild(s);
    console.log('[iframe] injected scroll script into parent head');
})();
</script>
""", height=1)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div id="home">', unsafe_allow_html=True)

# ── Particle network background ───────────────────────────────────────────────
components.html("""
<style>
#particle-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    opacity: 0.45;
}
</style>
<canvas id="particle-canvas"></canvas>
<script>
(function() {
    var canvas = document.getElementById('particle-canvas');

    // Move canvas to parent document
    var pdoc = window.parent.document;
    var pbody = pdoc.body;

    // Remove old canvas if exists
    var old = pdoc.getElementById('particle-canvas');
    if (old) old.remove();

    var c = pdoc.createElement('canvas');
    c.id = 'particle-canvas';
    c.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;opacity:0.4;';
    pbody.appendChild(c);

    var ctx = c.getContext('2d');
    var W, H, particles;
    var PARTICLE_COUNT = 80;
    var MAX_DIST = 140;
    var ACCENT = '74, 158, 255';  // matches --accent-primary blue

    function resize() {
        W = c.width  = window.parent.innerWidth;
        H = c.height = window.parent.innerHeight;
    }

    function Particle() {
        this.x  = Math.random() * W;
        this.y  = Math.random() * H;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.r  = Math.random() * 2 + 1;
    }

    function init() {
        resize();
        particles = [];
        for (var i = 0; i < PARTICLE_COUNT; i++) particles.push(new Particle());
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);

        // Update + draw dots
        for (var i = 0; i < particles.length; i++) {
            var p = particles[i];
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > W) p.vx *= -1;
            if (p.y < 0 || p.y > H) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(' + ACCENT + ', 0.8)';
            ctx.fill();
        }

        // Draw connecting lines
        for (var i = 0; i < particles.length; i++) {
            for (var j = i + 1; j < particles.length; j++) {
                var dx   = particles[i].x - particles[j].x;
                var dy   = particles[i].y - particles[j].y;
                var dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < MAX_DIST) {
                    var alpha = (1 - dist / MAX_DIST) * 0.5;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = 'rgba(' + ACCENT + ',' + alpha + ')';
                    ctx.lineWidth   = 1;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }

    window.parent.addEventListener('resize', function() { resize(); init(); });
    init();
    draw();
})();
</script>
""", height=0)

col_hero, col_shield = st.columns([6, 4])

with col_hero:
    st.markdown("""
<h1 style="font-size:3rem;font-weight:800;line-height:1.1;margin-bottom:1rem;
           background:linear-gradient(135deg,var(--accent-primary) 0%,var(--accent-secondary) 100%);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
  AI-Powered Phishing Website Detection
</h1>
<p style="font-size:1.1rem;color:var(--text-secondary);margin-bottom:2rem;">
  Detect malicious websites instantly using advanced ensemble machine learning
  with 99.65% ROC-AUC accuracy.
</p>
""", unsafe_allow_html=True)

    url_input = st.text_input(
        "Website URL", placeholder="Enter website URL  (e.g. https://example.com)",
        label_visibility="collapsed", key="url_input",
    )
    scan_btn  = st.button("🔍 Analyze Website", type="primary", use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, txt in zip([c1, c2, c3], ["✔ Real-time", "✔ 99.65% ROC-AUC", "✔ Hybrid ML"]):
        col.markdown(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'color:var(--text-secondary);font-size:14px;">{txt}</span>',
            unsafe_allow_html=True
        )

with col_shield:
    shield_slot = st.empty()
    with shield_slot.container():
        show_shield("neutral")

st.markdown("</div>", unsafe_allow_html=True)

# ── Analysis result ───────────────────────────────────────────────────────────
if scan_btn and url_input:
    # ── Input validation ──────────────────────────────────────────────────────
    if len(url_input) > 2000:
        st.error("❌ URL too long. Maximum 2000 characters.")
        st.stop()

    # Auto-prepend https if missing
    if not url_input.startswith(("http://", "https://")):
        url_input = "https://" + url_input

    # ── Rate limiting — max 1 scan every 3 seconds per session ───────────────
    import time as _time
    _now = _time.time()
    _last = st.session_state.get("last_scan_time", 0)
    if _now - _last < 3:
        st.warning("⏳ Please wait a moment before scanning again.")
        st.stop()
    st.session_state["last_scan_time"] = _now

    try:
        with st.status("🔍 Analyzing website…", expanded=True) as status:
            status.update(label="📊 Extracting URL features…",  state="running"); time.sleep(0.3)
            status.update(label="🌐 Fetching webpage content…", state="running"); time.sleep(0.3)
            status.update(label="🔎 Looking up WHOIS data…",      state="running"); time.sleep(0.2)
            status.update(label="🔒 Checking SSL certificate…",  state="running"); time.sleep(0.2)
            status.update(label="🛡️ Checking Google Safe Browsing…", state="running"); time.sleep(0.2)
            status.update(label="🤖 Running ML models…",        state="running")
            result = analyze_url(url_input, models, use_pilwd=use_pilwd)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)

        # ── Trusted domain whitelist (compute early for history + display) ──
        _TRUSTED_DOMAINS = {
            "google.com","www.google.com","mail.google.com","drive.google.com",
            "youtube.com","www.youtube.com","github.com","www.github.com",
            "microsoft.com","www.microsoft.com","outlook.com","live.com",
            "facebook.com","www.facebook.com","twitter.com","www.twitter.com",
            "x.com","www.x.com","linkedin.com","www.linkedin.com",
            "apple.com","www.apple.com","icloud.com",
            "amazon.com","www.amazon.com","aws.amazon.com",
            "wikipedia.org","www.wikipedia.org","stackoverflow.com",
            "www.stackoverflow.com","reddit.com","www.reddit.com",
            "netflix.com","www.netflix.com","bbc.com","www.bbc.com",
            "bbc.co.uk","www.bbc.co.uk","cnn.com","www.cnn.com",
            "instagram.com","www.instagram.com","whatsapp.com","www.whatsapp.com",
            "streamlit.io","www.streamlit.io","anthropic.com","www.anthropic.com",
            # Nigerian domains
            "mtn.com.ng","www.mtn.com.ng","gtbank.com","www.gtbank.com",
            "zenithbank.com","www.zenithbank.com","accessbankplc.com","www.accessbankplc.com",
            "firstbanknigeria.com","www.firstbanknigeria.com","uba.com","www.uba.com",
            "nigeriaairways.com","airpeace.com","www.airpeace.com",
            "dstv.com","www.dstv.com","multichoice.com","www.multichoice.com",
            "ncc.gov.ng","www.ncc.gov.ng","nimc.gov.ng","www.nimc.gov.ng",
            "biu.edu.ng","www.biu.edu.ng",
        }
        from urllib.parse import urlparse as _urlparse
        _ph_val      = result["phishing_prob"]
        _content_ok  = result.get("content_fetched", False)
        _ph_thresh   = 50 if _content_ok else 95
        _parsed_host = _urlparse(url_input).netloc.lower()
        # Match exact host, www-stripped host, or any subdomain of a trusted domain
        def _check_trusted(host, domains):
            if host in domains: return True
            bare = host.lstrip("www.")
            if bare in domains: return True
            # Check if host ends with .trustedomain (subdomain match)
            for d in domains:
                if host.endswith('.' + d): return True
            return False
        _is_trusted  = _check_trusted(_parsed_host, _TRUSTED_DOMAINS)
        adjusted_pred = 0 if _is_trusted else (1 if _ph_val >= _ph_thresh else 0)

        with shield_slot.container():
            show_shield("phishing" if adjusted_pred == 1 else "safe")

        # Save to scan history (keep last 5)
        st.session_state.scan_history.insert(0, {
            "url":        url_input,
            "verdict":    "Phishing" if adjusted_pred == 1 else "Safe",
            "confidence": round(result["confidence"], 1),
            "ph":         round(_ph_val, 1),
            "time":       pd.Timestamp.now().strftime("%H:%M:%S"),
        })
        st.session_state.scan_history = st.session_state.scan_history[:5]

        # Anchor for auto-scroll
        st.markdown('<div id="result-section"></div>', unsafe_allow_html=True)

        col_res, col_tips = st.columns([2, 1])

        with col_res:
            ph  = result["phishing_prob"]
            lg  = result["legit_prob"]
            cf  = result["confidence"]

            # Apply whitelist + threshold overrides
            if _is_trusted:
                ph = 0.0; lg = 100.0; cf = 100.0
                result["phishing_prob"] = 0.0
                result["legit_prob"]    = 100.0
                result["confidence"]    = 100.0
            else:
                ph = result["phishing_prob"]
                lg = result["legit_prob"]
                cf = result["confidence"]

            if adjusted_pred == 1:
                ver = "PHISHING DETECTED"
                ico = "⚠️"
                border_col = "var(--danger)"
                glow_col   = "rgba(255,61,0,0.25)"
            else:
                ver = "SAFE"
                ico = "✅"
                border_col = "var(--success)"
                glow_col   = "rgba(0,200,83,0.25)"

            # Show warning banner if content couldn't be fetched
            if not _content_ok:
                st.markdown("""
<div style="background:rgba(255,179,0,0.1);border:1px solid rgba(255,179,0,0.4);
     border-radius:10px;padding:10px 16px;margin-bottom:12px;
     display:flex;align-items:center;gap:10px;">
  <span style="font-size:18px;">⚠️</span>
  <span style="font-size:13px;color:#FFB300;">
    <b>URL-only analysis</b> — webpage content could not be fetched.
    Result is based on 25/38 features and may be less accurate.
    Confidence threshold raised to 95% to reduce false positives.
  </span>
</div>""", unsafe_allow_html=True)

            # ── Threat level gauge ─────────────────────────────────────────
            # Use adjusted prediction for gauge coloring
            if ph < 20:
                threat_label = "LOW RISK";      threat_col = "#00C853"; threat_icon = "🟢"
            elif ph < 50:
                threat_label = "MEDIUM RISK";   threat_col = "#FFB300"; threat_icon = "🟡"
            elif ph < 75:
                threat_label = "HIGH RISK";     threat_col = "#FF6D00"; threat_icon = "🟠"
            else:
                threat_label = "CRITICAL";      threat_col = "#FF3D00"; threat_icon = "🔴"

            # Gauge marker position (0–100% across the bar)
            marker_pos = ph

            st.markdown(f"""
<style>
@keyframes needleSlide {{
  from {{ left: 0%; }}
  to   {{ left: {marker_pos:.1f}%; }}
}}
@keyframes fadeSlideUp {{
  from {{ opacity:0; transform:translateY(20px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
.result-square {{ animation: fadeSlideUp 0.7s ease forwards; }}
#pg-needle     {{ animation: needleSlide 1.8s cubic-bezier(.22,.61,.36,1) 0.6s both; }}
#pg-pill-ph    {{ animation: fadeSlideUp 0.6s ease 1.2s both; }}
#pg-pill-lg    {{ animation: fadeSlideUp 0.6s ease 1.5s both; }}
</style>

<div class="result-square" style="border-color:{border_col};
     box-shadow:0 30px 40px -20px rgba(0,0,0,0.8), 0 0 0 1px {glow_col} inset;">
  <div style="font-size:3rem;margin-bottom:.5rem;">{ico}</div>
  <div style="font-size:2rem;font-weight:800;margin-bottom:.3rem;">{ver}</div>
  <div style="font-size:1rem;color:var(--text-secondary);margin-bottom:1.2rem;">
    Confidence: <span id="pg-cf">0.0</span>% &nbsp;·&nbsp; {result['model_used']}
  </div>

  <!-- Threat Level Gauge -->
  <div style="margin:0 0 20px 0;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span style="font-size:13px;color:var(--text-secondary);text-transform:uppercase;
                   letter-spacing:1px;">Threat Level</span>
      <span style="font-size:15px;font-weight:800;color:{threat_col};">
        {threat_icon} {threat_label}
      </span>
    </div>
    <!-- Rainbow spectrum bar -->
    <div style="position:relative;width:100%;height:12px;border-radius:999px;
         background:linear-gradient(90deg,#00C853 0%,#FFB300 40%,#FF6D00 65%,#FF3D00 100%);
         border:1px solid rgba(255,255,255,0.1);">
      <!-- Marker needle — starts at 0, animates to final position -->
      <div id="pg-needle" style="position:absolute;top:50%;left:{marker_pos:.1f}%;
           transform:translate(-50%,-50%);
           width:18px;height:18px;border-radius:50%;
           background:{threat_col};
           border:3px solid #fff;
           box-shadow:0 0 8px {threat_col};">
      </div>
    </div>
    <!-- Labels under bar -->
    <div style="display:flex;justify-content:space-between;margin-top:5px;">
      <span style="font-size:11px;color:#00C853;">Low</span>
      <span style="font-size:11px;color:#FFB300;">Medium</span>
      <span style="font-size:11px;color:#FF6D00;">High</span>
      <span style="font-size:11px;color:#FF3D00;">Critical</span>
    </div>
  </div>

  <div class="result-pills">
    <span id="pg-pill-ph" class="pill phishing">Phishing: <span id="pg-ph">{ph:.1f}</span>%</span>
    <span id="pg-pill-lg" class="pill legitimate">Legitimate: <span id="pg-lg">{lg:.1f}</span>%</span>
  </div>

  <div style="display:flex;gap:20px;margin-top:24px;">
    <div style="flex:1;text-align:center;padding:12px;
         background:var(--bg-primary);border-radius:12px;border:1px solid var(--border-color);">
      <div style="font-size:22px;font-weight:800;color:var(--accent-primary);">{cf:.1f}%</div>
      <div style="color:var(--text-secondary);font-size:13px;text-transform:uppercase;
                  letter-spacing:1px;">Confidence</div>
    </div>
    <div style="flex:1;text-align:center;padding:12px;
         background:var(--bg-primary);border-radius:12px;border:1px solid var(--border-color);">
      <div style="font-size:22px;">{'✅' if result['content_fetched'] else '⚠️'}</div>
      <div style="color:var(--text-secondary);font-size:13px;text-transform:uppercase;
                  letter-spacing:1px;">Content Fetched</div>
    </div>
    <div style="flex:1;text-align:center;padding:12px;
         background:var(--bg-primary);border-radius:12px;border:1px solid var(--border-color);">
      <div style="font-size:16px;font-weight:800;
           color:{'#00C853' if result['whois']['age_days'] and result['whois']['age_days'] > 365
                  else '#FFB300' if result['whois']['age_days'] and result['whois']['age_days'] > 30
                  else '#FF3D00'};">
        {result['whois']['age_str']}
      </div>
      <div style="color:var(--text-secondary);font-size:13px;text-transform:uppercase;
                  letter-spacing:1px;">Domain Age</div>
      <div style="color:var(--text-secondary);font-size:11px;margin-top:3px;">
        {result['whois']['creation_date']}
      </div>
    </div>
  </div>
  <!-- Registrar row -->
  <div style="margin-top:12px;padding:10px 14px;background:var(--bg-primary);
       border-radius:10px;border:1px solid var(--border-color);
       display:flex;justify-content:space-between;align-items:center;">
    <span style="color:var(--text-secondary);font-size:12px;text-transform:uppercase;
                 letter-spacing:1px;">Registrar</span>
    <span style="color:var(--text-primary);font-size:13px;font-weight:600;">
      {result['whois']['registrar']}
    </span>
  </div>
  <!-- Google Safe Browsing row -->
  <div style="margin-top:8px;padding:10px 14px;background:var(--bg-primary);
       border-radius:10px;border:1px solid var(--border-color);">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="display:flex;align-items:center;gap:6px;
                   color:var(--text-secondary);font-size:12px;text-transform:uppercase;letter-spacing:1px;">
        <span style="font-size:15px;">🛡️</span> Google Safe Browsing
      </span>
      <span style="font-size:13px;font-weight:700;
           color:{'#FF3D00' if result['gsb']['flagged'] == True else '#00C853' if result['gsb']['flagged'] == False else '#FFB300'};">
        {'🚨 FLAGGED' if result['gsb']['flagged'] == True else '✅ Not Listed' if result['gsb']['flagged'] == False else '⚠️ Unavailable'}
      </span>
    </div>
    {f'<div style="margin-top:4px;font-size:12px;color:#FF3D00;font-weight:600;">Threat: {result["gsb"]["threat"]}</div>' if result["gsb"]["flagged"] else ""}
  </div>
  <!-- SSL row -->
  <div style="margin-top:8px;padding:10px 14px;background:var(--bg-primary);
       border-radius:10px;border:1px solid var(--border-color);">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="color:var(--text-secondary);font-size:12px;text-transform:uppercase;
                   letter-spacing:1px;">SSL Certificate</span>
      <span style="font-size:13px;font-weight:700;
           color:{'#00C853' if result['ssl']['valid'] == True else '#FF3D00' if result['ssl']['valid'] == False else '#FFB300'};">
        {'🔒 Valid' if result['ssl']['valid'] == True else '🔓 Invalid' if result['ssl']['valid'] == False else '⚠️ Unavailable'}
      </span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:6px;">
      <span style="color:var(--text-secondary);font-size:12px;">
        Issuer: <span style="color:var(--text-primary);">{result['ssl']['issuer']}</span>
      </span>
      <span style="color:var(--text-secondary);font-size:12px;">
        Expires: <span style="color:{'#00C853' if result['ssl']['days_left'] and result['ssl']['days_left'] > 30 else '#FF3D00'};">
          {result['ssl']['expires']}{f" ({result['ssl']['days_left']}d)" if result['ssl']['days_left'] else ""}
        </span>
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Count-up + auto-scroll via components.html (scripts work here)
            components.html(f"""
<script>
(function() {{
    var doc = window.parent.document;
    var win = window.parent;
    var TARGET_PH = {ph:.2f};
    var TARGET_LG = {lg:.2f};
    var TARGET_CF = {cf:.2f};
    var DURATION  = 1800;
    var start     = null;

    function ease(t) {{ return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }}

    function animate(ts) {{
        if (!start) start = ts;
        var t = Math.min((ts - start) / DURATION, 1);
        var e = ease(t);
        var elPh = doc.getElementById('pg-ph');
        var elLg = doc.getElementById('pg-lg');
        var elCf = doc.getElementById('pg-cf');
        if (elPh) elPh.textContent = (TARGET_PH * e).toFixed(1);
        if (elLg) elLg.textContent = (TARGET_LG * e).toFixed(1);
        if (elCf) elCf.textContent = (TARGET_CF * e).toFixed(1);
        if (t < 1) requestAnimationFrame(animate);
    }}

    function scrollToResult() {{
        var target = doc.getElementById('result-section');
        if (!target) {{ setTimeout(scrollToResult, 100); return; }}
        var el = target.parentElement;
        while (el) {{
            var st = win.getComputedStyle(el);
            var ov = st.overflow + st.overflowY;
            if (/auto|scroll/.test(ov) && el.scrollHeight > el.clientHeight + 5) {{
                var dest = el.scrollTop + target.getBoundingClientRect().top
                           - el.getBoundingClientRect().top - 100;
                el.scrollTo({{ top: dest, behavior: 'smooth' }});
                break;
            }}
            el = el.parentElement;
        }}
        win.scrollTo({{ top: win.pageYOffset + target.getBoundingClientRect().top - 100, behavior: 'smooth' }});
    }}

    setTimeout(scrollToResult, 200);
    setTimeout(function() {{ requestAnimationFrame(animate); }}, 600);

    // ── Toast notification ─────────────────────────────────────────────
    function showToast() {{
        var doc  = window.parent.document;
        var old  = doc.getElementById('pg-toast');
        if (old) old.remove();

        var toast = doc.createElement('div');
        toast.id        = 'pg-toast';
        toast.className = 'pg-toast {("phishing" if adjusted_pred == 1 else "safe")}';
        toast.innerHTML = '{("⚠️ Phishing Detected!" if adjusted_pred == 1 else "✅ Site Looks Safe!")} &nbsp;<span style="font-weight:400;font-size:13px;">Confidence: {cf:.1f}%</span>';
        doc.body.appendChild(toast);

        setTimeout(function() {{ toast.classList.add('show'); }}, 50);
        setTimeout(function() {{
            toast.classList.remove('show');
            setTimeout(function() {{ toast.remove(); }}, 400);
        }}, 4000);
    }}
    setTimeout(showToast, 800);
}})();
</script>
""", height=0)

            st.markdown("#### 📥 Export Results")
            safe_name = (url_input.replace("https://","").replace("http://","")
                                  .replace("/","_").replace("?","_"))[:30]

            txt_report = (
                f"PHISHGUARD SECURITY REPORT\n{'='*40}\n"
                f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"URL       : {url_input}\n"
                f"VERDICT   : {ver} {ico}\n"
                f"Confidence: {cf:.1f}%\n"
                f"Model     : {result['model_used']}\n\n"
                f"Phishing risk  : {ph:.1f}%\n"
                f"Legitimate     : {lg:.1f}%\n"
                f"Content fetched: {'Yes' if result['content_fetched'] else 'No'}\n"
                f"{'='*40}\nReport by PhishGuard AI – Ernest Chimezulem Nna\n"
            )
            json_report = json.dumps({
                "generated": str(pd.Timestamp.now()),
                "url": url_input,
                "verdict": ver,
                "is_phishing": bool(adjusted_pred == 1),
                "confidence_pct": round(cf, 2),
                "phishing_pct":   round(ph, 2),
                "legitimate_pct": round(lg, 2),
                "model": result["model_used"],
                "content_fetched": result["content_fetched"],
                "features": {
                    k: (float(v) if isinstance(v, (int, float, np.integer, np.floating))
                        else str(v))
                    for k, v in result["features"].items() if k != "url"
                },
            }, indent=2)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("📄 Download TXT", txt_report,
                                   f"phishguard_{safe_name}.txt", "text/plain",
                                   use_container_width=True, key="dl_txt")
            with dl2:
                st.download_button("📊 Download JSON", json_report,
                                   f"phishguard_{safe_name}.json", "application/json",
                                   use_container_width=True, key="dl_json")

        with col_tips:
            if adjusted_pred == 1:
                st.error("⚠️ **Safety Tips**")
                st.markdown("""
- Do **not** enter personal info
- Do **not** download any files
- Close this tab immediately
- Report to IT / authorities
""")
            else:
                st.success("✅ **Safe Practices**")
                st.markdown("""
- Always verify HTTPS padlock
- Check for spelling in domain
- Don't reuse passwords
- Enable 2FA when available
""")

        if not result["content_fetched"]:
            st.warning("⚠️ Could not fetch webpage content. "
                       "Analysis is based on URL features only.")

        with st.expander("🔍 View Detailed Extracted Features"):
            tab_url, tab_cnt = st.tabs(["URL Features", "Content Features"])
            with tab_url:
                st.json({k: v for k, v in result["features"].items()
                         if k in URL_FEATURE_COLS})
            with tab_cnt:
                st.json({k: v for k, v in result["features"].items()
                         if k in CONTENT_FEATURE_COLS})

        # ── SHAP explanation ──────────────────────────────────────────────────
        with st.expander("🧠 Why did the model decide this? (SHAP Explanation)"):
            try:
                import shap, numpy as np, matplotlib.pyplot as plt
                from matplotlib.colors import LinearSegmentedColormap

                # Build feature vector
                feat_vals = np.array([result["features"].get(f, 0)
                                      for f in ALL_FEATURE_COLS], dtype=float).reshape(1,-1)
                feat_scaled = models["original_scaler"].transform(feat_vals)

                # Use RF_PILWD if available, else first estimator of voting ensemble
                if models.get("_pilwd_ok"):
                    rf_model = models["pilwd_models"]["RF_PILWD"]
                else:
                    rf_model = models["original_voting"].estimators_[0]
                explainer  = shap.TreeExplainer(rf_model)
                shap_vals  = explainer.shap_values(feat_scaled, check_additivity=False)

                # Extract SHAP values — shape is (1, 38, 2): [samples, features, classes]
                import numpy as np
                arr = np.array(shap_vals)
                if arr.ndim == 3:
                    sv = arr[0, :, 1]          # phishing class
                elif arr.ndim == 2:
                    sv = arr[0]
                elif isinstance(shap_vals, list):
                    raw = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                    sv = np.array(raw).flatten()
                else:
                    sv = arr.flatten()

                # Top 12 features by absolute SHAP value
                indices = np.argsort(np.abs(sv))[-12:]
                top_features = [ALL_FEATURE_COLS[i].replace("_", " ").title() for i in indices]
                top_shap     = sv[indices]

                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#1C2333")
                ax.set_facecolor("#1C2333")

                colors = ["#00C853" if v < 0 else "#FF3D00" for v in top_shap]
                bars   = ax.barh(top_features, top_shap, color=colors, height=0.6)

                ax.axvline(0, color="#555", linewidth=1)
                ax.set_xlabel("SHAP Value  (← Safe  |  Phishing →)", color="#aaa", fontsize=11)
                ax.tick_params(colors="white", labelsize=10)
                ax.set_title("Feature Contributions to This Prediction", color="white",
                             fontsize=13, fontweight="bold", pad=12)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a3a55")

                # Value labels on bars
                for bar, val in zip(bars, top_shap):
                    x = bar.get_width()
                    ax.text(x + (0.001 if x >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                            f"{val:+.3f}", va="center",
                            ha="left" if x >= 0 else "right",
                            color="white", fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("""
<p style="font-size:12px;color:#888;margin-top:8px;">
  🔴 Red bars = features pushing toward <b style="color:#FF3D00;">Phishing</b> &nbsp;|&nbsp;
  🟢 Green bars = features pushing toward <b style="color:#00C853;">Safe</b>
</p>""", unsafe_allow_html=True)

            except ImportError:
                st.info("📦 Install SHAP to enable this feature: `pip install shap`")
            except Exception as shap_err:
                st.warning(f"SHAP explanation unavailable: {shap_err}")

        # ── Model Comparison ──────────────────────────────────────────────────
        if models.get("_orig_ok") and models.get("_pilwd_ok"):
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.markdown('<h3 style="font-size:1.1rem;font-weight:700;margin-bottom:14px;">⚖️ Model Comparison</h3>',
                        unsafe_allow_html=True)

            # Run both models on same features
            feat_df = pd.DataFrame([result["features"]])
            if _is_trusted:
                orig_pred,  orig_proba  = 0, np.array([1.0, 0.0])
                pilwd_pred, pilwd_proba = 0, np.array([1.0, 0.0])
            else:
                orig_pred, orig_proba = _predict_original(models, feat_df.copy())
                pilwd_pred, pilwd_proba = _predict_pilwd(models, feat_df.copy())

            def _verdict_html(pred, proba, label):
                is_ph     = pred == 1
                verdict   = "PHISHING" if is_ph else "SAFE"
                icon      = "⚠️" if is_ph else "✅"
                color     = "#FF3D00" if is_ph else "#00C853"
                bg        = "rgba(255,61,0,0.08)" if is_ph else "rgba(0,200,83,0.08)"
                border    = "rgba(255,61,0,0.3)"  if is_ph else "rgba(0,200,83,0.3)"
                conf      = max(proba) * 100
                ph_prob   = proba[1] * 100
                lg_prob   = proba[0] * 100
                return f"""
<div style="background:{bg};border:1px solid {border};border-radius:14px;padding:20px;height:100%;">
  <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;
              color:var(--text-secondary);margin-bottom:8px;">{label}</div>
  <div style="font-size:22px;font-weight:800;color:{color};margin-bottom:12px;">
    {icon} {verdict}
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <div style="flex:1;background:var(--bg-primary);border-radius:8px;padding:10px;text-align:center;">
      <div style="font-size:18px;font-weight:700;color:{color};">{conf:.1f}%</div>
      <div style="font-size:11px;color:var(--text-secondary);">Confidence</div>
    </div>
    <div style="flex:1;background:var(--bg-primary);border-radius:8px;padding:10px;text-align:center;">
      <div style="font-size:18px;font-weight:700;color:#FF3D00;">{ph_prob:.1f}%</div>
      <div style="font-size:11px;color:var(--text-secondary);">Phishing</div>
    </div>
    <div style="flex:1;background:var(--bg-primary);border-radius:8px;padding:10px;text-align:center;">
      <div style="font-size:18px;font-weight:700;color:#00C853;">{lg_prob:.1f}%</div>
      <div style="font-size:11px;color:var(--text-secondary);">Safe</div>
    </div>
  </div>
</div>"""

            col_orig, col_pilwd = st.columns(2)
            with col_orig:
                st.markdown(_verdict_html(orig_pred, orig_proba,
                             "Original Ensemble (RF + XGB)"),
                             unsafe_allow_html=True)
            with col_pilwd:
                st.markdown(_verdict_html(pilwd_pred, pilwd_proba,
                             "PILWD Ensemble (RF + XGB + GB)"),
                             unsafe_allow_html=True)

            # Agreement indicator
            agree      = orig_pred == pilwd_pred
            agr_color  = "#00C853" if agree else "#FFB300"
            agr_icon   = "✅" if agree else "⚠️"
            agr_text   = "Both models agree" if agree else "Models disagree — treat with caution"
            st.markdown(f"""
<div style="margin-top:12px;text-align:center;font-size:13px;
            color:{agr_color};font-weight:600;">
  {agr_icon} {agr_text}
</div>""", unsafe_allow_html=True)

    except Exception as exc:
        import traceback
        st.error(f"❌ Error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

# ── Scan History ─────────────────────────────────────────────────────────────
if st.session_state.get("scan_history"):
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:1.2rem;font-weight:700;margin-bottom:12px;">🕒 Recent Scans</h3>',
                unsafe_allow_html=True)
    for item in st.session_state.scan_history:
        is_ph   = item["verdict"] == "Phishing"
        v_color = "#FF3D00" if is_ph else "#00C853"
        v_icon  = "⚠️" if is_ph else "✅"
        st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:10px 16px;margin-bottom:8px;border-radius:10px;
     background:var(--bg-secondary);border:1px solid var(--border-color);
     border-left:3px solid {v_color};">
  <div style="display:flex;align-items:center;gap:10px;overflow:hidden;">
    <span style="font-size:18px;">{v_icon}</span>
    <span style="font-size:13px;color:var(--text-primary);white-space:nowrap;
                 overflow:hidden;text-overflow:ellipsis;max-width:320px;">{item["url"]}</span>
  </div>
  <div style="display:flex;align-items:center;gap:16px;flex-shrink:0;">
    <span style="font-size:13px;font-weight:700;color:{v_color};">{item["verdict"]}</span>
    <span style="font-size:12px;color:var(--text-secondary);">{item["confidence"]}%</span>
    <span style="font-size:12px;color:var(--text-secondary);">{item["time"]}</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Why Phishing Matters ──────────────────────────────────────────────────────
st.markdown("""
<div class="why-grid">
  <div class="card">
    <div class="card-icon">🎯</div>
    <div class="card-title">Increasing Phishing Attacks</div>
    <div class="card-description">Billions are lost yearly due to deceptive websites
      and fraudulent login pages.</div>
  </div>
  <div class="card">
    <div class="card-icon">🔗</div>
    <div class="card-title">Sophisticated URL Manipulation</div>
    <div class="card-description">Attackers hide behind misleading domain names and
      complex URL structures.</div>
  </div>
  <div class="card">
    <div class="card-icon">🧠</div>
    <div class="card-title">Need for Intelligent Detection</div>
    <div class="card-description">Traditional blacklist systems fail to detect newly
      created phishing websites.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── How It Works ──────────────────────────────────────────────────────────────
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">How Our Hybrid Detection System Works</h2>',
            unsafe_allow_html=True)
st.markdown("""
<div class="steps-grid">
  <div class="step-card">
    <div class="step-number">1</div>
    <div class="step-title">URL Feature Extraction</div>
    <div class="step-details">
      · Special characters analysis<br>· Subdomain count<br>
      · Domain &amp; path length<br>· Suspicious keyword patterns
    </div>
  </div>
  <div class="step-card">
    <div class="step-number">2</div>
    <div class="step-title">Content Feature Analysis</div>
    <div class="step-details">
      · HTML structure scanning<br>· Form &amp; password field detection<br>
      · JavaScript redirect patterns<br>· Brand keyword presence
    </div>
  </div>
  <div class="step-card">
    <div class="step-number">3</div>
    <div class="step-title">Hybrid ML Classification</div>
    <div class="step-details">
      · Random Forest model<br>· XGBoost model<br>
      · Soft-voting ensemble<br>· Confidence scoring
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Performance Metrics ───────────────────────────────────────────────────────
st.markdown('<div id="performance"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">System Performance</h2>',
            unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl, note in zip(
    [m1, m2, m3, m4],
    ["99.65%", "96.7%", "2.2%", "3"],
    ["ROC-AUC Score", "Phishing Detection", "False Positive Rate", "Ensemble Models"],
    ['<span style="color:#00C853;display:inline-block;transform:scaleX(1.8);font-weight:900;">↑</span> PILWD Ensemble',
     '<span style="color:#00C853;display:inline-block;transform:scaleX(1.8);font-weight:900;">↑</span> +2.7%',
     '<span style="color:#FF3D00;display:inline-block;transform:scaleX(1.8);font-weight:900;">↓</span> −1.3%',
     "RF + XGB + GB"],
):
    col.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{val}</div>
  <div class="metric-label">{lbl}</div>
  <div style="font-size:13px;margin-top:6px;">{note}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Feature Importance ────────────────────────────────────────────────────────
if models.get("_pilwd_ok"):
    st.markdown('<h2 class="section-title">Top Detection Features (PILWD)</h2>',
                unsafe_allow_html=True)

    rf_m  = models["pilwd_models"]["RF_PILWD"]
    f_ns  = models["pilwd_feat_names"]
    imp   = (pd.DataFrame({"feature": f_ns, "importance": rf_m.feature_importances_})
               .sort_values("importance", ascending=False).head(10))
    cats  = {"URL": "#ff6b6b", "Content": "#4ecdc4", "Other": "#95e1d3"}
    imp["cat"] = imp["feature"].apply(
        lambda f: "URL" if f.startswith("U") else ("Content" if f.startswith("H") else "Other")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1C2333"); ax.set_facecolor("#1C2333")
    ax.barh(range(len(imp)), imp["importance"],
            color=[cats[c] for c in imp["cat"]])
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp["feature"], color="white")
    ax.set_xlabel("Importance", color="white"); ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_edgecolor("#30363D")
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in cats.items()],
              facecolor="#1C2333", labelcolor="white")
    st.pyplot(fig); plt.close(fig)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Global Phishing Threat Map ────────────────────────────────────────────────
st.markdown('''<div style="text-align:center;margin-bottom:30px;">
  <h2 class="section-title" style="margin-bottom:6px;">Global Phishing Landscape</h2>
  <p style="color:var(--text-secondary);font-size:14px;margin:0;">
    Top 5 countries by phishing attack origin &nbsp;·&nbsp; Source: Kaspersky Threat Intelligence Report 2024
  </p>
</div>''', unsafe_allow_html=True)

components.html("""
<style>
  body { margin:0; background:transparent; }
  #map-container {
    width: 100%; height: 480px;
    background: #060d1a;
    border-radius: 14px;
    position: relative;
    overflow: hidden;
  }
  canvas#dotmap { display:block; }
  .map-legend {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 14px;
    padding: 0 10px;
    font-family: Inter, sans-serif;
    background: transparent;
  }
  .legend-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; color: #aaa;
    white-space: nowrap;
  }
  .legend-dot { width:10px;height:10px;border-radius:50%;background:#FF3D00;box-shadow:0 0 6px #FF3D00;flex-shrink:0; }
  .legend-pct { color:#FF6D00;font-weight:700; }
</style>

<div id="map-container">
  <canvas id="dotmap"></canvas>
</div>

<div class="map-legend">
  <div class="legend-item"><div class="legend-dot"></div><img src="https://flagcdn.com/16x12/ru.png" style="width:16px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px;">Russia <span class="legend-pct">&nbsp;~30%</span></div>
  <div class="legend-item"><div class="legend-dot"></div><img src="https://flagcdn.com/16x12/cn.png" style="width:16px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px;">China <span class="legend-pct">&nbsp;~14%</span></div>
  <div class="legend-item"><div class="legend-dot"></div><img src="https://flagcdn.com/16x12/us.png" style="width:16px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px;">USA <span class="legend-pct">&nbsp;~11%</span></div>
  <div class="legend-item"><div class="legend-dot"></div><img src="https://flagcdn.com/16x12/br.png" style="width:16px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px;">Brazil <span class="legend-pct">&nbsp;~8%</span></div>
  <div class="legend-item"><div class="legend-dot"></div><img src="https://flagcdn.com/16x12/ng.png" style="width:16px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px;">Nigeria <span class="legend-pct">&nbsp;~6%</span></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js"></script>
<script>
(function() {
  // ISO numeric codes for highlighted countries
  var HIGHLIGHT = {
    '643': { name:'Russia',  pct:'~30%', flag:'🇷🇺', cc:'ru', order:1 },
    '156': { name:'China',   pct:'~14%', flag:'🇨🇳', cc:'cn', order:2 },
    '840': { name:'USA',     pct:'~11%', flag:'🇺🇸', cc:'us', order:0 },
    '076': { name:'Brazil',  pct:'~8%',  flag:'🇧🇷', cc:'br', order:4 },
    '566': { name:'Nigeria', pct:'~6%',  flag:'🇳🇬', cc:'ng', order:3 },
  };
  // Preload flag images for canvas drawing
  var FLAG_IMGS = {};
  Object.values(HIGHLIGHT).forEach(function(hl) {
    var img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = 'https://flagcdn.com/16x12/' + hl.cc + '.png';
    FLAG_IMGS[hl.cc] = img;
  });
  // Also try alternate IDs for Brazil and USA in case topojson differs
  var HIGHLIGHT_ALT = { '76':'076', '840':'840' };

  var container = document.getElementById('map-container');
  var canvas    = document.getElementById('dotmap');
  var W = container.clientWidth;
  var H = container.clientHeight;
  canvas.width  = W;
  canvas.height = H;
  var ctx = canvas.getContext('2d');

  var DOT   = 3;    // dot radius
  var GAP   = 7;    // spacing between dots
  var STEP  = DOT * 2 + GAP;

  fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
    .then(function(r) { return r.json(); })
    .then(function(world) {
      var countries = topojson.feature(world, world.objects.countries);

      var projection = d3.geoNaturalEarth1()
        .scale(W / 6.3)
        .translate([W / 2, H / 2]);

      var path = d3.geoPath().projection(projection);

      // Build offscreen canvas per country for hit-testing
      function isInCountry(feature, px, py) {
        var oc = document.createElement('canvas');
        oc.width = W; oc.height = H;
        var oc2 = oc.getContext('2d');
        var p2  = d3.geoPath().projection(projection).context(oc2);
        oc2.beginPath();
        p2(feature);
        return oc2.isPointInPath(px, py);
      }

      // Pre-render each highlight country mask
      var hlMasks = {};
      Object.keys(HIGHLIGHT).forEach(function(id) {
        var feat = countries.features.find(function(f) {
          var fid = String(f.id).padStart(3,'0');
          return fid === id || String(f.id) === id;
        });
        if (!feat) return;
        var oc = document.createElement('canvas');
        oc.width = W; oc.height = H;
        var oc2 = oc.getContext('2d');
        var p2  = d3.geoPath().projection(projection).context(oc2);
        oc2.beginPath(); p2(feat); oc2.closePath();
        oc2.fill();
        hlMasks[id] = oc2;
      });

      // Pre-render all-world mask
      var worldCanvas = document.createElement('canvas');
      worldCanvas.width = W; worldCanvas.height = H;
      var wCtx = worldCanvas.getContext('2d');
      var wp   = d3.geoPath().projection(projection).context(wCtx);
      wCtx.beginPath(); wp({type:'FeatureCollection', features: countries.features});
      wCtx.closePath(); wCtx.fill();

      // Draw dots
      ctx.clearRect(0, 0, W, H);

      for (var x = STEP; x < W; x += STEP) {
        for (var y = STEP; y < H; y += STEP) {
          // Check world
          var wPx = wCtx.getImageData(x, y, 1, 1).data;
          if (wPx[3] < 10) continue; // not on land

          // Check if highlighted
          var hlId = null;
          for (var id in hlMasks) {
            var px = hlMasks[id].getImageData(x, y, 1, 1).data;
            if (px[3] > 10) { hlId = id; break; }
          }

          if (hlId) {
            // Highlighted country — bigger, glowing red dot
            ctx.beginPath();
            ctx.arc(x, y, DOT + 1.5, 0, Math.PI * 2);
            ctx.fillStyle = '#FF3D00';
            ctx.shadowColor = '#FF3D00';
            ctx.shadowBlur  = 8;
            ctx.fill();
            ctx.shadowBlur = 0;
          } else {
            // Regular country — small dim dot
            ctx.beginPath();
            ctx.arc(x, y, DOT - 0.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(74,120,180,0.45)';
            ctx.fill();
          }
        }
      }

      // Draw country labels directly on canvas (name + pct, no emoji)
      ctx.shadowBlur = 0;
      countries.features.forEach(function(feat) {
        var id  = String(feat.id);
        var id3 = id.padStart(3,'0');
        var hl  = HIGHLIGHT[id] || HIGHLIGHT[id3];
        if (!hl) return;
        var centroid = path.centroid(feat);
        if (!centroid || isNaN(centroid[0])) return;
        var cx = centroid[0], cy = centroid[1] - 18;
        var label = hl.name + ' ' + hl.pct;
        var flagImg = FLAG_IMGS[hl.cc];
        var flagW = (flagImg && flagImg.complete && flagImg.naturalWidth) ? 16 : 0;
        var flagGap = flagW ? flagW + 4 : 0;
        ctx.font = 'bold 11px Inter, sans-serif';
        var tw = ctx.measureText(label).width + flagGap;
        ctx.fillStyle   = 'rgba(6,13,26,0.88)';
        ctx.strokeStyle = 'rgba(255,61,0,0.7)';
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.roundRect(cx - tw/2 - 6, cy - 13, tw + 12, 20, 4);
        ctx.fill(); ctx.stroke();
        if (flagW) { ctx.drawImage(flagImg, cx - tw/2, cy - 10, 16, 12); }
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, cx - tw/2 + flagGap, cy);
      });

      // Animate pulsing rings on highlighted centroids
      var rings = [];
      countries.features.forEach(function(feat) {
        var id = String(feat.id);
        if (!HIGHLIGHT[id]) return;
        var c = path.centroid(feat);
        if (!c || isNaN(c[0])) return;
        rings.push({ x: c[0], y: c[1], r: 0, alpha: 0.8,
                     delay: Math.random() * 2000 });
      });

      // Build centroids for highlighted countries
      var centroids = {};
      countries.features.forEach(function(feat) {
        var id  = String(feat.id);
        var id3 = id.padStart(3,'0');  // normalise to 3 digits
        var hl  = HIGHLIGHT[id] || HIGHLIGHT[id3];
        if (!hl) return;
        var c = path.centroid(feat);
        if (!c || isNaN(c[0])) return;
        centroids[id3] = c;
      });

      // Resolve Brazil — try both '76' and '076'
      if (!centroids['076']) centroids['076'] = centroids['76'] || null;
      if (!centroids['76'])  centroids['76']  = centroids['076'] || null;

      // Chain: USA -> Russia -> China -> Nigeria -> Brazil -> USA (closed loop)
      var chainOrder = [
        ['840','643'],
        ['643','156'],
        ['156','566'],
        ['566','076'],
        ['076','840'],  // Brazil back to USA — closes the loop
      ];

      var connections = [];
      chainOrder.forEach(function(pair, i) {
        var fromId = pair[0], toId = pair[1];
        var from   = centroids[fromId];
        var to     = centroids[toId];
        if (!from || !to) { console.log('missing centroid', fromId, toId); return; }
        connections.push({
          from:     from,
          to:       to,
          progress: i * 0.2,
          speed:    0.00025,
        });
      });

      // Pre-compute control points for each arc
      connections.forEach(function(conn) {
        var x1 = conn.from[0], y1 = conn.from[1];
        var x2 = conn.to[0],   y2 = conn.to[1];
        conn.mx = (x1+x2)/2;
        conn.my = (y1+y2)/2 - Math.hypot(x2-x1, y2-y1) * 0.38;
      });

      // Draw arcs on main canvas (called every frame, before dots)
      function drawArcs() {
        connections.forEach(function(conn) {
          ctx.beginPath();
          ctx.moveTo(conn.from[0], conn.from[1]);
          ctx.quadraticCurveTo(conn.mx, conn.my, conn.to[0], conn.to[1]);
          ctx.strokeStyle = 'rgba(255,120,40,0.5)';
          ctx.lineWidth   = 1.5;
          ctx.setLineDash([5, 8]);
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }

      // Draw moving dot along arc
      function drawTraveller(conn) {
        var t  = conn.progress;
        var x1 = conn.from[0], y1 = conn.from[1];
        var x2 = conn.to[0],   y2 = conn.to[1];
        var tx = (1-t)*(1-t)*x1 + 2*(1-t)*t*conn.mx + t*t*x2;
        var ty = (1-t)*(1-t)*y1 + 2*(1-t)*t*conn.my + t*t*y2;
        ctx.beginPath();
        ctx.arc(tx, ty, 4, 0, Math.PI*2);
        ctx.fillStyle   = '#FF9500';
        ctx.shadowColor = '#FF6D00';
        ctx.shadowBlur  = 14;
        ctx.fill();
        ctx.shadowBlur  = 0;
      }

      // Dot rendering helper (reused each frame)
      function redrawDots() {
        for (var x = STEP; x < W; x += STEP) {
          for (var y = STEP; y < H; y += STEP) {
            var wPx = wCtx.getImageData(x, y, 1, 1).data;
            if (wPx[3] < 10) continue;
            var hlId2 = null;
            for (var hid in hlMasks) {
              if (hlMasks[hid].getImageData(x,y,1,1).data[3] > 10) { hlId2 = hid; break; }
            }
            if (hlId2) {
              ctx.beginPath(); ctx.arc(x,y,DOT+1.5,0,Math.PI*2);
              ctx.fillStyle='#FF3D00'; ctx.shadowColor='#FF3D00'; ctx.shadowBlur=8;
              ctx.fill(); ctx.shadowBlur=0;
            } else {
              ctx.beginPath(); ctx.arc(x,y,DOT-0.5,0,Math.PI*2);
              ctx.fillStyle='rgba(74,120,180,0.45)'; ctx.fill();
            }
          }
        }
      }

      var lastTs = null;
      function animateRings(ts) {
        if (!lastTs) lastTs = ts;
        var dt = ts - lastTs; lastTs = ts;
        ctx.clearRect(0, 0, W, H);

        // 1. Dots
        redrawDots();

        // 2. Static arc lines
        drawArcs();

        // 3. Moving dots
        connections.forEach(function(conn) {
          conn.progress += conn.speed * dt;
          if (conn.progress > 1) conn.progress = 0;
          drawTraveller(conn);
        });

        // 4. Pulse rings
        rings.forEach(function(ring) {
          ring.delay -= dt;
          if (ring.delay > 0) return;
          ring.r    += dt * 0.04;
          ring.alpha = Math.max(0, 0.7 - ring.r / 35);
          if (ring.r > 35) { ring.r = 0; ring.alpha = 0.7; ring.delay = Math.random()*1500; }
          ctx.beginPath();
          ctx.arc(ring.x, ring.y, ring.r, 0, Math.PI*2);
          ctx.strokeStyle = 'rgba(255,61,0,' + ring.alpha + ')';
          ctx.lineWidth   = 2;
          ctx.stroke();
        });

        // 5. Labels — redrawn each frame so animation doesn't erase them
        countries.features.forEach(function(feat) {
          var id  = String(feat.id);
          var id3 = id.padStart(3,'0');
          var hl  = HIGHLIGHT[id] || HIGHLIGHT[id3];
          if (!hl) return;
          var centroid = path.centroid(feat);
          if (!centroid || isNaN(centroid[0])) return;
          var cx = centroid[0], cy = centroid[1] - 18;
          var label = hl.name + ' ' + hl.pct;
          var flagImg = FLAG_IMGS[hl.cc];
          var flagW = (flagImg && flagImg.complete && flagImg.naturalWidth) ? 16 : 0;
          var flagGap = flagW ? flagW + 4 : 0;
          ctx.font = 'bold 11px Inter, sans-serif';
          var tw = ctx.measureText(label).width + flagGap;
          ctx.fillStyle   = 'rgba(6,13,26,0.88)';
          ctx.strokeStyle = 'rgba(255,61,0,0.7)';
          ctx.lineWidth   = 1; ctx.beginPath();
          ctx.roundRect(cx - tw/2 - 6, cy - 13, tw + 12, 20, 4);
          ctx.fill(); ctx.stroke();
          if (flagW) {
            ctx.drawImage(flagImg, cx - tw/2, cy - 10, 16, 12);
          }
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, cx - tw/2 + flagGap, cy);
        });

        requestAnimationFrame(animateRings);
      }
      requestAnimationFrame(animateRings);
    })
    .catch(function(e) {
      ctx.fillStyle = '#aaa';
      ctx.font = '14px Inter';
      ctx.fillText('Map unavailable — check internet connection', 40, H/2);
    });
})();
</script>
""", height=560)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Comparison ────────────────────────────────────────────────────────────────
st.markdown('<h2 class="section-title">What Makes This System Different</h2>',
            unsafe_allow_html=True)
st.markdown("""
<div class="comparison-grid">
  <div class="comparison-card">
    <div class="comparison-title">Traditional Systems</div>
    <ul class="comparison-list">
      <li class="comparison-item"><span>✗</span> Blacklist dependent</li>
      <li class="comparison-item"><span>✗</span> No hybrid analysis</li>
      <li class="comparison-item"><span>✗</span> Limited features</li>
      <li class="comparison-item"><span>✗</span> Poor new-attack detection</li>
    </ul>
  </div>
  <div class="comparison-card highlight">
    <div class="comparison-title">Our Hybrid System</div>
    <ul class="comparison-list">
      <li class="comparison-item"><span>✓</span> URL + Content integration</li>
      <li class="comparison-item"><span>✓</span> Multiple ML models</li>
      <li class="comparison-item"><span>✓</span> Real-time detection</li>
      <li class="comparison-item"><span>✓</span> Higher accuracy</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Evaluation Page ───────────────────────────────────────────────────────────
st.markdown('<div id="evaluation"></div>', unsafe_allow_html=True)
st.markdown('''<div style="text-align:center;margin-bottom:24px;">
  <h2 class="section-title" style="margin-bottom:8px;">Model Evaluation</h2>
  <p style="color:var(--text-secondary);font-size:14px;margin:0;">
    Performance metrics evaluated on 20% holdout test set from combined_urls.csv
  </p>
</div>''', unsafe_allow_html=True)

@st.cache_data(show_spinner="Running model evaluation…")
def run_evaluation():
    import pandas as pd, numpy as np, pickle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, confusion_matrix, roc_auc_score, roc_curve)
    from src.url_features import extract_url_features, URL_FEATURE_COLS

    # Load dataset
    df = pd.read_csv("dataset/combined_urls.csv").dropna(subset=["url","label"])
    df = df.sample(min(10000, len(df)), random_state=42)  # cap at 10k for speed

    # Extract URL features only (content features are 0 during training)
    feat_df = extract_url_features(df[["url"]])
    for col in URL_FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0
    X = feat_df[URL_FEATURE_COLS].fillna(0).values
    y = df["label"].values

    # Load scaler + model
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    model  = pickle.load(open("models/voting_model.pkl", "rb"))

    # Scale (only URL features — pad content cols with zeros)
    import joblib
    full_scaler = pickle.load(open("models/scaler.pkl","rb"))
    n_expected  = full_scaler.n_features_in_
    n_url       = X.shape[1]
    if n_expected > n_url:
        X = np.hstack([X, np.zeros((X.shape[0], n_expected - n_url))])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = full_scaler.transform(X_test)

    y_pred  = model.predict(X_test_sc)
    y_prob  = model.predict_proba(X_test_sc)[:,1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred)    * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred)        * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob)   * 100, 2),
    }
    cm  = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return metrics, cm, fpr.tolist(), tpr.tolist()

st.markdown("""
<style>
.eval-btn-wrap button {
    background: linear-gradient(135deg,#4A9EFF,#0066CC) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 10px 32px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 0 16px rgba(74,158,255,0.3) !important;
}
.eval-btn-wrap button p {
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="eval-btn-wrap">', unsafe_allow_html=True)
    run_eval = st.button("📊 Run Evaluation", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
if run_eval:
    try:
        metrics, cm, fpr, tpr = run_evaluation()

        # ── Metric cards ─────────────────────────────────────────────────────
        cols = st.columns(5)
        colors = ["#4A9EFF","#00C853","#FFB300","#FF6D00","#9C27B0"]
        for col, (label, val), color in zip(cols, metrics.items(), colors):
            col.markdown(f"""
<div class="metric-card" style="border-top:3px solid {color};text-align:center;padding:16px;">
  <div style="font-size:28px;font-weight:800;color:{color};">{val}%</div>
  <div style="font-size:12px;color:var(--text-secondary);text-transform:uppercase;
              letter-spacing:1px;margin-top:4px;">{label}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2)

        # ── Confusion Matrix ──────────────────────────────────────────────────
        with col_cm:
            st.markdown('<h3 style="text-align:center;margin-bottom:12px;">Confusion Matrix</h3>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#1C2333")
            ax.set_facecolor("#1C2333")
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Legitimate","Phishing"], color="white")
            ax.set_yticklabels(["Legitimate","Phishing"], color="white")
            ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
            ax.tick_params(colors="white")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                            fontsize=16, fontweight="bold",
                            color="white" if cm[i,j] > cm.max()/2 else "black")
            for s in ax.spines.values(): s.set_edgecolor("#30363D")
            plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="white")
            st.pyplot(fig); plt.close(fig)

        # ── ROC Curve ────────────────────────────────────────────────────────
        with col_roc:
            st.markdown('<h3 style="text-align:center;margin-bottom:12px;">ROC Curve</h3>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#1C2333")
            ax.set_facecolor("#1C2333")
            ax.plot(fpr, tpr, color="#4A9EFF", linewidth=2.5,
                    label=f'AUC = {metrics["ROC-AUC"]}%')
            ax.plot([0,1],[0,1], color="#555", linestyle="--", linewidth=1)
            ax.fill_between(fpr, tpr, alpha=0.1, color="#4A9EFF")
            ax.set_xlabel("False Positive Rate", color="white")
            ax.set_ylabel("True Positive Rate",  color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1C2333", labelcolor="white", fontsize=11)
            for s in ax.spines.values(): s.set_edgecolor("#30363D")
            st.pyplot(fig); plt.close(fig)

    except Exception as e:
        import traceback
        st.error(f"Evaluation error: {e}")
        with st.expander("Traceback"): st.code(traceback.format_exc())
st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── About ─────────────────────────────────────────────────────────────────────
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">About This Project</h2>',
            unsafe_allow_html=True)
st.markdown("""
<div class="about-card">
  <p class="about-text">
    This Hybrid Machine Learning-Based Phishing Website Detection System was developed to
    enhance cybersecurity awareness and automate malicious website detection using advanced
    URL and content feature analysis. The system combines Random Forest and XGBoost models
    for optimal detection accuracy. The latest enhancement incorporates the PILWD dataset
    ensemble, achieving 99.65% ROC-AUC.
  </p>
  <div class="about-details">
    <div class="about-item">
      <div class="about-label">Developer</div>
      <div class="about-value">Ernest Chimezulem Nna</div>
    </div>
    <div class="about-item">
      <div class="about-label">Department</div>
      <div class="about-value">Computer Science</div>
    </div>
    <div class="about-item">
      <div class="about-label">University</div>
      <div class="about-value">Benson Idahosa University</div>
    </div>
    <div class="about-item">
      <div class="about-label">Year</div>
      <div class="about-value">2026</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
# ── Footer + back-to-top ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <p class="footer-text">
    © 2026 PhishGuard AI &nbsp;·&nbsp;
    <span class="footer-name">Ernest Chimezulem Nna</span>
  </p>
  <p class="footer-text" style="font-size:12px;">
    Powered by Hybrid Ensemble ML &nbsp;·&nbsp; 99.65% ROC-AUC
  </p>
</div>
<button onclick="window.scrollTo({top:0,behavior:'smooth'});"
        id="backToTop" class="back-to-top" title="Back to top">↑</button>
""", unsafe_allow_html=True)

# ── JS: scroll progress + back-to-top ────────────────────────────────────────
st.markdown("""
<script>
(function () {
    function init() {
        var bar = document.getElementById('pgbar');
        var btn = document.getElementById('backToTop');
        if (!bar) return;
        function onScroll() {
            var scrolled = document.documentElement.scrollTop || document.body.scrollTop;
            var total    = document.documentElement.scrollHeight
                         - document.documentElement.clientHeight;
            bar.style.width = (total > 0 ? (scrolled / total) * 100 : 0) + '%';
            if (btn) btn.classList.toggle('show', scrolled > 300);
        }
        window.addEventListener('scroll', onScroll);
        onScroll();
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', function () { setTimeout(init, 400); });
    else
        setTimeout(init, 400);
})();
</script>
""", unsafe_allow_html=True)