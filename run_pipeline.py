# run_pipeline.py
"""
CogniShield End-to-End Orchestration Pipeline.
Generates data → Extracts features → Trains models → Computes SHAP attributions
→ Scores risks → Persists to SQLite + CSV.
"""
import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.models import (
    train_autoencoder,
    compute_autoencoder_scores,
    compute_autoencoder_feature_deltas,
    compute_lstm_sequence_scores,
    compute_isolation_forest_scores,
    compute_shap_attributions,
)
from src.risk_engine import CogniShieldRiskEngine
from src.database import init_database, store_features, store_threat_feed

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "user_daily_features.csv"
OUTPUT_FEED_PATH = PROJECT_ROOT / "data" / "processed" / "threat_intelligence_feed.csv"
DATABASE_PATH = PROJECT_ROOT / "data" / "cognishield.db"
RANDOM_SEED = 42


def main():
    print("==========================================================")
    print(" COGNISHIELD: INSIDER THREAT DETECTION & RISK INTELLIGENCE")
    print("==========================================================")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    features_path = FEATURES_PATH
    if not features_path.exists():
        print(f"[-] Error: {features_path} not found. Run src/feature_extraction.py first.")
        return

    # 0. Initialize SQLite Database
    print("[*] Initializing SQLite database layer...")
    db_path = init_database(str(DATABASE_PATH))

    # 1. Load Preprocessed Telemetry
    df = pd.read_csv(features_path)
    print(f"[*] Ingested {len(df)} user-day activity vectors across {df['user'].nunique()} entities.")

    feature_cols = [
        'total_logons', 'total_logoffs', 'after_hours_logons', 'weekend_logons',
        'usb_connects', 'files_accessed', 'files_copied_external'
    ]

    # Store features in SQLite
    store_features(df, db_path)
    print(f"[+] Feature matrix persisted to SQLite ({db_path})")

    # 2. Normalize Numerical Matrix
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    # 3. Deep Autoencoder (Reconstruction Anomaly)
    print("[*] Training Deep Autoencoder on baseline behavioral profiles...")
    ae_model = train_autoencoder(X_scaled, epochs=50, batch_size=16, lr=0.005)
    ae_scores = compute_autoencoder_scores(ae_model, X_scaled)

    # 3b. Compute per-feature AE reconstruction deltas (XAI)
    print("[*] Computing Autoencoder per-feature reconstruction deltas...")
    ae_deltas = compute_autoencoder_feature_deltas(ae_model, X_scaled)

    # 4. Temporal Sequence Modeling (LSTM)
    print("[*] Running Temporal LSTM sequence deviation analysis...")
    lstm_scores = np.zeros(len(df))
    for user_id in df['user'].unique():
        idx = df[df['user'] == user_id].index
        user_matrix = X_scaled[idx]
        user_lstm = compute_lstm_sequence_scores(user_matrix, seq_len=3)
        lstm_scores[idx] = user_lstm

    # 5. Isolation Forest (Boundary Anomaly Detection)
    print("[*] Evaluating tabular boundaries via Isolation Forest...")
    if_scores, iso_model = compute_isolation_forest_scores(X_scaled, contamination=0.08)

    # 5b. Compute SHAP feature attributions for Isolation Forest (XAI)
    print("[*] Computing SHAP TreeExplainer attributions for Isolation Forest...")
    shap_values = compute_shap_attributions(iso_model, X_scaled)

    # 6. Dynamic Risk Scoring Engine with Feature Attributions
    print("[*] Synthesizing behavioral, temporal, and lexical scores...")
    risk_engine = CogniShieldRiskEngine(w_ae=0.35, w_lstm=0.25, w_if=0.20, w_nlp=0.20)

    # Compute unified feature attributions
    nlp_scores = df['lexical_risk_score'].to_numpy()
    attributions = risk_engine.compute_feature_attributions(
        shap_values=shap_values,
        ae_deltas=ae_deltas,
        ae_scores=ae_scores,
        if_scores=if_scores,
        lstm_scores=lstm_scores,
        nlp_scores=nlp_scores,
        feature_names=feature_cols
    )

    scored_df = risk_engine.compute_risk_profiles(
        df, ae_scores, lstm_scores, if_scores,
        shap_attributions=attributions
    )

    # 7. Persist Structured Threat Intelligence Feed
    output_feed = OUTPUT_FEED_PATH
    scored_df.to_csv(output_feed, index=False)
    print(f"[+] Risk intelligence feed exported to CSV ({output_feed})")

    # Persist to SQLite
    store_threat_feed(scored_df, db_path)
    print(f"[+] Risk intelligence feed persisted to SQLite ({db_path})")

    # Summary of High Risk Alerts
    high_risks = scored_df[scored_df['risk_level'] == 'High']
    print("\n----------------------------------------------------------")
    print(f"[!] THREAT ASSESSMENT SUMMARY: {len(high_risks)} High Risk Incident(s) Detected")
    print("----------------------------------------------------------")
    if not high_risks.empty:
        summary_cols = ['date', 'user', 'risk_score', 'risk_level', 'files_copied_external', 'flagged_terms']
        print(high_risks[summary_cols].to_string(index=False))

        # Display attribution breakdown for top incident
        top_incident = high_risks.iloc[0]
        print(f"\n[*] XAI Attribution for {top_incident['user']} ({top_incident['date']}):")
        import json
        attrs = json.loads(top_incident['shap_attributions'])
        sorted_attrs = sorted(attrs.items(), key=lambda x: x[1], reverse=True)
        for feat, pts in sorted_attrs:
            if pts > 0.5:
                print(f"     - {feat}: +{pts:.1f} pts")

    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    main()