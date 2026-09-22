# src/feature_extraction.py
import os
import re
import pandas as pd
import numpy as np

SUSPICIOUS_KEYWORDS = [
    r'confidential', r'proprietary', r'salary', r'leak',
    r'resignation', r'bypass', r'admin pass', r'backup', r'dump'
]

def analyze_nlp_intent(text_series: pd.Series) -> pd.DataFrame:
    """
    Computes lexical risk scores and suspicious term counts from communication telemetry.
    """
    scores = []
    term_counts = []
    
    for text in text_series.fillna(""):
        lower_text = str(text).lower()
        matches = [k for k in SUSPICIOUS_KEYWORDS if re.search(k, lower_text)]
        count = len(matches)
        score = min(1.0, count * 0.35)
        scores.append(score)
        term_counts.append(count)
        
    return pd.DataFrame({'lexical_risk_score': scores, 'flagged_terms': term_counts})

def build_feature_matrix(raw_dir="data/raw", output_path="data/processed/user_daily_features.csv"):
    """
    Aggregates logon, device, file, and text logs into normalized daily user vectors.
    """
    print("Ingesting raw data feeds...")
    logon_df = pd.read_csv(f"{raw_dir}/logon.csv")
    device_df = pd.read_csv(f"{raw_dir}/device.csv")
    file_df = pd.read_csv(f"{raw_dir}/file.csv")
    text_df = pd.read_csv(f"{raw_dir}/text.csv")

    # 1. Process Logon & Temporal Activity
    logon_df['timestamp'] = pd.to_datetime(logon_df['timestamp'])
    logon_df['date'] = logon_df['timestamp'].dt.date.astype(str)
    logon_df['hour'] = logon_df['timestamp'].dt.hour
    logon_df['is_weekend'] = logon_df['timestamp'].dt.weekday >= 5
    logon_df['is_after_hours'] = (logon_df['hour'] < 7) | (logon_df['hour'] > 19)

    logon_agg = logon_df.groupby(['date', 'user']).agg(
        total_logons=('activity', lambda x: (x == 'Logon').sum()),
        total_logoffs=('activity', lambda x: (x == 'Logoff').sum()),
        after_hours_logons=('is_after_hours', lambda x: (x & (logon_df.loc[x.index, 'activity'] == 'Logon')).sum()),
        weekend_logons=('is_weekend', lambda x: (x & (logon_df.loc[x.index, 'activity'] == 'Logon')).sum())
    ).reset_index()

    # 2. Process Peripheral Device (USB) Activity
    device_df['timestamp'] = pd.to_datetime(device_df['timestamp'])
    device_df['date'] = device_df['timestamp'].dt.date.astype(str)
    device_agg = device_df.groupby(['date', 'user']).agg(
        usb_connects=('activity', lambda x: (x == 'Connect').sum())
    ).reset_index()

    # 3. Process File Access & Exfiltration Actions
    file_df['timestamp'] = pd.to_datetime(file_df['timestamp'])
    file_df['date'] = file_df['timestamp'].dt.date.astype(str)
    file_agg = file_df.groupby(['date', 'user']).agg(
        files_accessed=('filename', 'count'),
        files_copied_external=('to_removable_media', 'sum')
    ).reset_index()

    # 4. Process Textual Telemetry via NLP
    text_df['timestamp'] = pd.to_datetime(text_df['timestamp'])
    text_df['date'] = text_df['timestamp'].dt.date.astype(str)
    nlp_results = analyze_nlp_intent(text_df['content'])
    text_df['lexical_risk_score'] = nlp_results['lexical_risk_score']
    text_df['flagged_terms'] = nlp_results['flagged_terms']

    text_agg = text_df.groupby(['date', 'user']).agg(
        lexical_risk_score=('lexical_risk_score', 'max'),
        flagged_terms=('flagged_terms', 'sum')
    ).reset_index()

    # 5. Multi-Source Fusion
    merged = pd.merge(logon_agg, device_agg, on=['date', 'user'], how='outer')
    merged = pd.merge(merged, file_agg, on=['date', 'user'], how='outer')
    merged = pd.merge(merged, text_agg, on=['date', 'user'], how='outer')

    # Impute missing counts with 0
    numeric_cols = [
        'total_logons', 'total_logoffs', 'after_hours_logons', 'weekend_logons',
        'usb_connects', 'files_accessed', 'files_copied_external',
        'lexical_risk_score', 'flagged_terms'
    ]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    # Sort chronological profile
    merged.sort_values(by=['user', 'date'], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Feature matrix successfully constructed: {merged.shape[0]} rows, {merged.shape[1]} features.")
    print(f"Saved to: {output_path}")
    return merged

if __name__ == "__main__":
    build_feature_matrix()