"""
URL Feature Extraction for Phishing Detection
25 features extracted purely from the URL string (no network calls).
"""

import re
import math
from urllib.parse import urlparse


def url_length(url):
    return len(url)

def dot_count(url):
    return url.count('.')

def hyphen_count(url):
    return url.count('-')

def at_symbol_count(url):
    return 1 if '@' in url else 0

def question_mark_count(url):
    return url.count('?')

def equal_count(url):
    return url.count('=')

def slash_count(url):
    return url.count('/')

def has_https(url):
    return 1 if url.startswith('https') else 0

def has_ip(url):
    return 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0

def suspicious_words(url):
    """Generic phishing keywords - no brand names."""
    keywords = [
        'login', 'signin', 'verify', 'confirm', 'account', 'update',
        'secure', 'password', 'credential', 'authenticate', 'banking',
        'suspend', 'restrict', 'unlock', 'alert', 'notification',
        'urgent', 'immediate', 'expire', 'limited', 'access',
        'security', 'validate', 'reverify', 'identity', 'invoice',
        'payment', 'billing', 'support', 'customer', 'service',
        'reward', 'winner', 'prize', 'gift', 'claim', 'free',
        'money', 'dollar', 'bitcoin', 'crypto', 'wallet',
        'admin', 'administrator', 'dashboard', 'panel',
        'webscr', 'cmd', 'token', 'session', 'oauth'
    ]
    url_lower = url.lower()
    return sum(1 for kw in keywords if kw in url_lower)

def subdomain_count(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain:
            parts = domain.split('.')
            if len(parts) > 2:
                return len(parts) - 2
    except Exception:
        pass
    return 0

def domain_length(url):
    try:
        parsed = urlparse(url)
        return len(parsed.netloc)
    except Exception:
        return 0

def path_length(url):
    try:
        return len(urlparse(url).path)
    except Exception:
        return 0

def query_length(url):
    try:
        return len(urlparse(url).query)
    except Exception:
        return 0

def num_digits(url):
    return sum(c.isdigit() for c in url)

def num_letters(url):
    return sum(c.isalpha() for c in url)

def num_special_chars(url):
    special = set('!#$%^&*()_+-=[]{}|;:,.<>?')
    return sum(1 for c in url if c in special)

def has_port(url):
    return 1 if re.search(r':\d{2,5}', url) else 0

def has_double_slash(url):
    # Ignore the protocol's //
    return 1 if '//' in url[8:] else 0

def url_entropy(url):
    if not url:
        return 0.0
    prob = [float(url.count(c)) / len(url) for c in set(url)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

def has_encoded_chars(url):
    return 1 if '%' in url and re.search(r'%[0-9A-Fa-f]{2}', url) else 0

def num_parameters(url):
    try:
        query = urlparse(url).query
        return len(query.split('&')) if query else 0
    except Exception:
        return 0

def tld_length(url):
    try:
        domain = urlparse(url).netloc
        if '.' in domain:
            return len(domain.split('.')[-1])
    except Exception:
        pass
    return 0

def https_count(url):
    return url.lower().count('https')

def digit_letter_ratio(url):
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    return digits / letters if letters > 0 else 0.0


# ── ordered list used everywhere ──────────────────────────────────────────────
URL_FEATURE_COLS = [
    'url_length', 'dots', 'hyphens', 'at_symbol', 'question_marks',
    'equals', 'slashes', 'https', 'ip', 'suspicious_words', 'subdomains',
    'domain_length', 'path_length', 'query_length', 'num_digits',
    'num_letters', 'special_chars', 'has_port', 'double_slash',
    'url_entropy', 'encoded_chars', 'num_parameters', 'tld_length',
    'https_count', 'digit_letter_ratio'
]


def extract_url_features(df):
    """
    Add all 25 URL features to *df* (which must have a 'url' column).
    Returns the same DataFrame with feature columns appended.
    """
    df = df.copy()
    df['url_length']        = df['url'].apply(url_length)
    df['dots']              = df['url'].apply(dot_count)
    df['hyphens']           = df['url'].apply(hyphen_count)
    df['at_symbol']         = df['url'].apply(at_symbol_count)
    df['question_marks']    = df['url'].apply(question_mark_count)
    df['equals']            = df['url'].apply(equal_count)
    df['slashes']           = df['url'].apply(slash_count)
    df['https']             = df['url'].apply(has_https)
    df['ip']                = df['url'].apply(has_ip)
    df['suspicious_words']  = df['url'].apply(suspicious_words)
    df['subdomains']        = df['url'].apply(subdomain_count)
    df['domain_length']     = df['url'].apply(domain_length)
    df['path_length']       = df['url'].apply(path_length)
    df['query_length']      = df['url'].apply(query_length)
    df['num_digits']        = df['url'].apply(num_digits)
    df['num_letters']       = df['url'].apply(num_letters)
    df['special_chars']     = df['url'].apply(num_special_chars)
    df['has_port']          = df['url'].apply(has_port)
    df['double_slash']      = df['url'].apply(has_double_slash)
    df['url_entropy']       = df['url'].apply(url_entropy)
    df['encoded_chars']     = df['url'].apply(has_encoded_chars)
    df['num_parameters']    = df['url'].apply(num_parameters)
    df['tld_length']        = df['url'].apply(tld_length)
    df['https_count']       = df['url'].apply(https_count)
    df['digit_letter_ratio']= df['url'].apply(digit_letter_ratio)
    return df