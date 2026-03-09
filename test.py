import streamlit as st
import pandas as pd
import pickle
from src.url_features import extract_url_features
import matplotlib.pyplot as plt
import numpy as np

# ===================== CONFIGURATION =====================
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern 3D styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
    
    /* Main Title with Glow Effect */
    .main-title {
        font-size: 56px;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(0, 210, 255, 0.5); }
        to { text-shadow: 0 0 40px rgba(0, 210, 255, 0.8), 0 0 60px rgba(58, 123, 213, 0.4); }
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 18px;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* 3D Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* Result Box with 3D Effect */
    .result-box {
        padding: 30px 50px;
        border-radius: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transform: perspective(1000px) rotateX(0deg);
        transition: all 0.5s ease;
    }
    
    .phishing {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border: 2px solid #ff6b6b;
    }
    
    .legitimate {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: 2px solid #38ef7d;
    }
    
    /* 3D Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Field */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: white;
        padding: 15px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
    }
    
    /* Metrics with 3D effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Feature cards */
    .feature-item {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateX(10px);
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%);
    }
    
    /* How it works steps */
    .step-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .step-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    /* Sample URL buttons */
    .sample-btn {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px 15px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .sample-btn:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Warning/Success boxes */
    .stWarning, .stSuccess {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== HELPER FUNCTIONS =====================

@st.cache_resource
def load_models():
    """Load trained models and scaler"""
    model = pickle.load(open('models/voting_model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    return model, scaler

def analyze_url(url, model, scaler):
    """Analyze a URL and return prediction with details"""
    df = pd.DataFrame({'url': [url]})
    df = extract_url_features(df)
    
    feature_cols = ['url_length', 'dots', 'hyphens', 'at_symbol', 'https', 'ip', 'suspicious_words']
    features = df[feature_cols]
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    confidence = max(probability) * 100
    phishing_prob = probability[1] * 100
    legit_prob = probability[0] * 100
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'phishing_prob': phishing_prob,
        'legit_prob': legit_prob,
        'features': df[feature_cols].iloc[0].to_dict()
    }

# ===================== SIDEBAR =====================

def render_sidebar():
    """Render modern sidebar with project info"""
    st.sidebar.markdown("""
    <style>
        .sidebar-title {
            font-size: 24px;
            font-weight: 700;
            color: #00d2ff;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    <div class="sidebar-title">🛡️ Phishing Detector</div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Model Info Card
    st.sidebar.markdown("""
    <div class="glass-card" style="padding: 20px;">
        <h3 style="color: #00d2ff; margin-bottom: 15px;">🤖 Model Info</h3>
        <p style="color: #ccc;"><strong>Algorithm:</strong> Voting Ensemble</p>
        <p style="color: #ccc; margin-bottom: 10px;"><strong>Components:</strong></p>
        <ul style="color: #aaa; padding-left: 20px;">
            <li>Random Forest</li>
            <li>XGBoost</li>
            <li>Soft Voting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Features Card
    st.sidebar.markdown("""
    <div class="glass-card" style="padding: 20px;">
        <h3 style="color: #38ef7d; margin-bottom: 15px;">📊 Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    features = [
        ("URL Length", "Total characters"),
        ("Dot Count", "Number of dots"),
        ("Hyphen Count", "Hyphens used"),
        ("@ Symbol", "Contains @"),
        ("HTTPS", "Security protocol"),
        ("IP Address", "IP in URL"),
        ("Suspicious Words", "Phishing keywords")
    ]
    
    for icon, name in features:
        st.sidebar.markdown(f"""
        <div class="feature-item">
            <span style="color: #00d2ff;">▸</span> {name}
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # About Card
    st.sidebar.markdown("""
    <div class="glass-card" style="padding: 20px;">
        <h3 style="color: #ff6b6b; margin-bottom: 10px;">ℹ️ About</h3>
        <p style="color: #aaa; font-size: 14px;">
            ML-powered phishing detection using hybrid ensemble methods.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <p style="text-align: center; color: #666;">
        🎓 Final Year Project 2024
    </p>
    """, unsafe_allow_html=True)

# ===================== MAIN PAGE =====================

def render_main_page():
    """Render the main page with modern 3D styling"""
    
    # Title
    st.markdown('<p class="main-title">🎣 Phishing URL Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Phishing Detection using Hybrid Machine Learning</p>', unsafe_allow_html=True)
    
    # Main Input Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.header("🔗 URL Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com", label_visibility="collapsed")

with col2:
    analyze_button = st.button("🔍 Analyze", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("##### 📝 Quick Test", unsafe_allow_html=True)

sample_col1, sample_col2, sample_col3 = st.columns(3)

sample_urls = {"Google": "https://www.google.com", "GitHub": "https://github.com", "Phishing": "http://www.phishtank.com/phish_detail.php?phish_id=7793166"}

with sample_col1:
    if st.button("✅ Google", use_container_width=True):
        url_input = sample_urls["Google"]
        st.rerun()
with sample_col2:
    if st.button("✅ GitHub", use_container_width=True):
        url_input = sample_urls["GitHub"]
        st.rerun()
with sample_col3:
    if st.button("⚠️ Phishing Test", use_container_width=True):
        url_input = sample_urls["Phishing"]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Analysis Section
if analyze_button and url_input:
    if not url_input.startswith(('http://', 'https://')):
        url_input = 'http://' + url_input
        st.info("🔗 Added 'http://' prefix")
    
    with st.spinner("🔄 Analyzing URL..."):
        try:
            model, scaler = load_models()
            result = analyze_url(url_input, model, scaler)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.header("📊 Results")
            
            if result['prediction'] == 1:
                st.markdown('<div class="result-box phishing">⚠️ PHISHING DETECTED!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box legitimate">✅ LEGITIMATE URL</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.markdown(f'<div class="metric-card"><p style="color: #aaa;">Confidence</p><h2 style="color: #00d2ff;">{result["confidence"]:.2f}%</h2></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><p style="color: #aaa;">Phishing</p><h2 style="color: #ff6b6b;">{result["phishing_prob"]:.2f}%</h2></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><p style="color: #aaa;">Legitimate</p><h2 style="color: #38ef7d;">{result["legit_prob"]:.2f}%</h2></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📈 Probability Distribution")
            
            prob_data = pd.DataFrame({'Category': ['Phishing', 'Legitimate'], 'Probability': [result['phishing_prob'], result['legit_prob']]})
            colors = ['#ff5252', '#4caf50']
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(prob_data['Category'], prob_data['Probability'], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.set_title('URL Classification Probability')
            for bar, prob in zip(bars, prob_data['Probability']):
                ax.text(prob + 2, bar.get_y() + bar.get_height()/2, f'{prob:.2f}%', va='center', fontweight='bold')
            st.pyplot(fig)
            
            # Feature breakdown
            st.subheader("🔍 Feature Analysis")
            fc1, fc2, fc3, fc4 = st.columns(4)
            feature_cols = ['url_length', 'dots', 'hyphens', 'at_symbol', 'https', 'ip', 'suspicious_words']
            feature_names = ['URL Length', 'Dots', 'Hyphens', '@ Symbol', 'HTTPS', 'IP Address', 'Suspicious']
            for i, (col, fname) in enumerate(zip([fc1, fc2, fc3, fc4, fc1, fc2, fc3], feature_names)):
                with col:
                    st.metric(label=fname, value=result['features'][feature_cols[i]])
            
            # Safety tips
            if result['prediction'] == 1:
                st.warning("⚠️ **Safety Tips:**\n- Do not enter personal information\n- Do not download files\n- Report this URL")
            else:
                st.success("✅ **Safe!** This URL appears legitimate.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif analyze_button and not url_input:
    st.warning("Please enter a URL")

# How It Works
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.header("💡 How It Works")

hw1, hw2, hw3, hw4 = st.columns(4)
steps = [
    ("1️⃣", "Input URL", "Enter any URL to check"),
    ("2️⃣", "Extract Features", "Analyze URL structure"),
    ("3️⃣", "ML Prediction", "Hybrid model analyzes"),
    ("4️⃣", "Get Result", "View prediction instantly")
]

for col, (icon, title, desc) in zip([hw1, hw2, hw3, hw4], steps):
    with col:
        st.markdown(f"**{icon} {title}**\n\n{desc}")

st.markdown('</div>', unsafe_allow_html=True)