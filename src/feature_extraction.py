# src/feature_extraction.py
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "user_daily_features.csv"

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

def _read_and_validate(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required telemetry feed not found: {path}")
    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} contains no telemetry rows")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    if frame["user"].isna().any() or frame["user"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path.name} contains blank user identifiers")
    frame["timestamp"] = timestamps.dt.tz_convert(None)
    return frame


def build_feature_matrix(raw_dir=DEFAULT_RAW_DIR, output_path=DEFAULT_OUTPUT_PATH):
    """
    Aggregates logon, device, file, and text logs into normalized daily user vectors.
    """
    print("Ingesting raw data feeds...")
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    logon_df = _read_and_validate(raw_dir / "logon.csv", {"timestamp", "user", "activity", "pc"})
    device_df = _read_and_validate(raw_dir / "device.csv", {"timestamp", "user", "activity"})
    file_df = _read_and_validate(raw_dir / "file.csv", {"timestamp", "user", "filename", "to_removable_media"})
    text_df = _read_and_validate(raw_dir / "text.csv", {"timestamp", "user", "content"})

    if not logon_df["activity"].isin({"Logon", "Logoff"}).all():
        raise ValueError("logon.csv contains unsupported activity values")
    if not device_df["activity"].isin({"Connect"}).all():
        raise ValueError("device.csv contains unsupported activity values")
    removable_values = pd.to_numeric(file_df["to_removable_media"], errors="coerce")
    if removable_values.isna().any() or ~removable_values.isin({0, 1}).all():
        raise ValueError("file.csv to_removable_media must contain only 0 or 1")
    file_df["to_removable_media"] = removable_values

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

    os.makedirs(output_path.parent, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Feature matrix successfully constructed: {merged.shape[0]} rows, {merged.shape[1]} features.")
    print(f"Saved to: {output_path}")
    return merged

if __name__ == "__main__":
    build_feature_matrix()