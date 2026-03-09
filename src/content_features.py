"""
Content Feature Extraction for Phishing Detection
Fetches a page and extracts 14 HTML/content-based features.

FIX: Previous version produced doubled column names like
     'content_content_num_forms' because extract_content_features()
     returned keys already prefixed with 'content_' AND the cache-building
     loop also added the prefix.  All keys are now consistently prefixed
     once, and content_feature_names() matches exactly.
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False


def _get_domain(url: str) -> str:
    """Extract registrable domain, works with or without tldextract."""
    if _HAS_TLDEXTRACT:
        ext = tldextract.extract(url)
        return ext.domain + '.' + ext.suffix
    parsed = urlparse(url)
    parts = parsed.netloc.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else parsed.netloc


# ── ordered list used everywhere ──────────────────────────────────────────────
CONTENT_FEATURE_COLS = [
    'content_num_forms',
    'content_has_password_field',
    'content_suspicious_form_action',
    'content_external_link_ratio',
    'content_has_js_redirect',
    'content_has_alert',
    'content_num_hidden_elements',
    'content_num_iframes',
    'content_brand_keyword_count',
    'content_suspicious_phrase_count',
    'content_has_title',
    'content_title_has_brand',
    'content_external_favicon',
    'content_fetched',          # 1 = page was reachable, 0 = fetch failed
]


def content_feature_names():
    """Return ordered list of content feature names."""
    return CONTENT_FEATURE_COLS.copy()


def _zero_features():
    """Return all-zero feature dict (used when fetch fails)."""
    return {col: 0 for col in CONTENT_FEATURE_COLS}


def fetch_page(url, timeout=10):
    """Fetch page HTML. Returns HTML string or None."""
    try:
        url = url.strip().replace(' ', '')
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        resp = requests.get(url, timeout=timeout, headers=headers,
                            allow_redirects=True)
        return resp.text if resp.status_code == 200 else None
    except Exception:
        return None


def extract_content_features(url, html=None):
    """
    Extract 14 content-based features from a webpage.
    If html is None the page is fetched automatically.
    Always returns a dict with exactly the keys in CONTENT_FEATURE_COLS.
    """
    if html is None:
        html = fetch_page(url)
    if html is None:
        return _zero_features()          # content_fetched = 0

    soup = BeautifulSoup(html, 'html.parser')
    feats = {}

    # ── Forms ─────────────────────────────────────────────────────────────────
    forms = soup.find_all('form')
    feats['content_num_forms'] = len(forms)
    feats['content_has_password_field'] = (
        1 if soup.find('input', {'type': 'password'}) else 0
    )

    src_domain = _get_domain(url)
    suspicious_action = 0
    for form in forms:
        action = form.get('action', '')
        if action and action.startswith('http'):
            act_domain = _get_domain(action)
            if act_domain != src_domain:
                suspicious_action = 1
        elif action and not action.startswith(('/', '#', '?')):
            suspicious_action = 1
    feats['content_suspicious_form_action'] = suspicious_action

    # ── Links ─────────────────────────────────────────────────────────────────
    all_links = soup.find_all('a', href=True)
    total_links = len(all_links)
    external = sum(
        1 for a in all_links
        if a['href'].startswith('http') and not a['href'].startswith(url)
    )
    feats['content_external_link_ratio'] = (
        external / total_links if total_links > 0 else 0
    )

    # ── JavaScript ────────────────────────────────────────────────────────────
    scripts = soup.find_all('script')
    js_redirect = re.compile(r'window\.location|document\.location|window\.open')
    feats['content_has_js_redirect'] = (
        1 if any(js_redirect.search(str(s)) for s in scripts) else 0
    )
    feats['content_has_alert'] = (
        1 if any('alert(' in str(s) for s in scripts) else 0
    )

    # ── Hidden elements & iframes ─────────────────────────────────────────────
    hidden = soup.find_all(
        style=re.compile(r'display:\s*none|visibility:\s*hidden')
    )
    feats['content_num_hidden_elements'] = len(hidden)
    feats['content_num_iframes'] = len(soup.find_all('iframe'))

    # ── Text / keyword features ───────────────────────────────────────────────
    page_text = soup.get_text().lower()
    brand_kw = [
        'paypal', 'apple', 'bank', 'verify', 'account',
        'update', 'secure', 'login', 'password'
    ]
    feats['content_brand_keyword_count'] = sum(
        page_text.count(kw) for kw in brand_kw
    )
    suspicious_phrases = [
        'verify your account', 'update your information',
        'confirm password', 'sign in to continue'
    ]
    feats['content_suspicious_phrase_count'] = sum(
        page_text.count(ph) for ph in suspicious_phrases
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    title = soup.find('title')
    feats['content_has_title'] = 1 if title else 0
    title_text = title.get_text().lower() if title else ''
    feats['content_title_has_brand'] = (
        1 if any(kw in title_text for kw in brand_kw) else 0
    )

    # ── Favicon ───────────────────────────────────────────────────────────────
    fav = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
    if fav and fav.get('href', '').startswith('http') \
            and not fav['href'].startswith(url):
        feats['content_external_favicon'] = 1
    else:
        feats['content_external_favicon'] = 0

    feats['content_fetched'] = 1
    return feats
