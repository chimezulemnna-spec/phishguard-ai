"""
Data loading and preprocessing for Phishing Detection.

FIX: legitimate_urls.csv contains bare domains (no protocol).
     We now add 'https://' so URL feature extraction works correctly.
"""

import pandas as pd


def _ensure_protocol(url: str) -> str:
    url = str(url).strip()
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url


def load_and_preprocess(phishing_path: str, legitimate_path: str) -> pd.DataFrame:
    """
    Load phishing + legitimate CSVs, label them, normalise URLs,
    and return a combined shuffled DataFrame with columns ['url', 'label'].
    """
    df_phish = pd.read_csv(phishing_path)
    df_legit = pd.read_csv(legitimate_path)

    # ── Phishing ──────────────────────────────────────────────────────────────
    df_phish = df_phish[['url']].copy()
    df_phish['label'] = 1

    # ── Legitimate ────────────────────────────────────────────────────────────
    # Handle multiple possible column names
    if 'url' in df_legit.columns:
        df_legit = df_legit[['url']].copy()
    elif 'URL' in df_legit.columns:
        df_legit = df_legit.rename(columns={'URL': 'url'})[['url']].copy()
    elif 'domain' in df_legit.columns:
        df_legit = df_legit.rename(columns={'domain': 'url'})[['url']].copy()
    else:
        raise ValueError(
            f"Cannot find URL column in {legitimate_path}. "
            f"Available: {df_legit.columns.tolist()}"
        )

    # FIX: add protocol to bare domains
    df_legit['url'] = df_legit['url'].apply(_ensure_protocol)
    df_legit['label'] = 0

    # ── Combine ───────────────────────────────────────────────────────────────
    df = pd.concat([df_phish, df_legit], ignore_index=True)
    df = df[['url', 'label']]
    df['url'] = df['url'].astype(str).str.strip()
    df = df.dropna(subset=['url'])
    df = df[df['url'] != '']
    df = df.drop_duplicates(subset=['url'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Loaded {len(df):,} URLs  "
          f"(phishing={df['label'].sum():,}, "
          f"legitimate={(df['label']==0).sum():,})")
    return df
