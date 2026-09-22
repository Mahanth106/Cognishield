# src/api.py
"""
CogniShield Threat Intelligence REST API.
Built with FastAPI — serves risk scores, alerts, user histories,
SHAP explainability, and mitigation triggers via structured endpoints.
Now backed by SQLite with CSV fallback for backward compatibility.
"""
import os
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="CogniShield Threat Intelligence API",
    description="REST API for batch insider threat results, explainable risk attribution, and user risk retrieval.",
    version="1.1.0"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "cognishield.db")
FEED_PATH = str(PROJECT_ROOT / "data" / "processed" / "threat_intelligence_feed.csv")


def get_telemetry_data():
    """Load threat intelligence data from SQLite (preferred) or CSV (fallback)."""
    # Prefer SQLite
    if os.path.exists(DB_PATH):
        from src.database import query_all_threats
        df = query_all_threats(DB_PATH)
        if not df.empty:
            return df

    # Fallback to CSV
    if not os.path.exists(FEED_PATH):
        raise HTTPException(
            status_code=404,
            detail="Threat intelligence feed data not found. Run pipeline first."
        )
    return pd.read_csv(FEED_PATH)


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "CogniShield Context-Aware Threat Engine",
        "version": "1.1.0",
        "database": "SQLite" if os.path.exists(DB_PATH) else "CSV"
    }


@app.get("/api/v1/alerts/high-risk")
def get_high_risk_alerts():
    """Returns all incidents flagged as High Risk."""
    df = get_telemetry_data()
    high_risks = df[df['risk_level'] == 'High']
    return {
        "count": len(high_risks),
        "alerts": high_risks.drop(columns=['shap_attributions'], errors='ignore').to_dict(orient="records")
    }


@app.get("/api/v1/user/{user_id}")
def get_user_risk_history(user_id: str):
    """Retrieves full longitudinal threat profile for a specific user entity."""
    df = get_telemetry_data()
    user_records = df[df['user'] == user_id.upper()]
    if user_records.empty:
        raise HTTPException(status_code=404, detail=f"User entity '{user_id}' not found.")

    latest = user_records.sort_values(by='date').iloc[-1]
    return {
        "user_id": user_id.upper(),
        "latest_risk_score": float(latest['risk_score']),
        "latest_risk_level": latest['risk_level'],
        "history": user_records.drop(columns=['shap_attributions'], errors='ignore').to_dict(orient="records")
    }


@app.get("/api/v1/user/{user_id}/explain")
def get_user_risk_explanation(user_id: str):
    """
    Returns SHAP feature attribution breakdown for a user's highest-risk assessment.
    Shows which behavioral features contributed most to the risk score.
    """
    df = get_telemetry_data()
    user_records = df[df['user'] == user_id.upper()]
    if user_records.empty:
        raise HTTPException(status_code=404, detail=f"User entity '{user_id}' not found.")

    # Get the highest-risk record
    top_record = user_records.sort_values(by='risk_score', ascending=False).iloc[0]

    if 'shap_attributions' not in top_record or pd.isna(top_record.get('shap_attributions')):
        raise HTTPException(
            status_code=404,
            detail="SHAP attributions not available. Re-run pipeline with Phase 1 updates."
        )

    attributions = json.loads(top_record['shap_attributions'])
    sorted_attrs = sorted(attributions.items(), key=lambda x: x[1], reverse=True)

    # Human-readable feature name mapping
    feature_labels = {
        'total_logons': 'Total Logon Events',
        'total_logoffs': 'Total Logoff Events',
        'after_hours_logons': 'After-Hours Access',
        'weekend_logons': 'Weekend Access',
        'usb_connects': 'USB Connections',
        'files_accessed': 'Files Accessed',
        'files_copied_external': 'External File Copies',
        'lstm_temporal': 'Temporal Sequence Deviation',
        'nlp_lexical': 'Lexical Threat Indicators'
    }

    explanation = []
    for feat, pts in sorted_attrs:
        explanation.append({
            "feature": feat,
            "label": feature_labels.get(feat, feat),
            "contribution_points": pts
        })

    return {
        "user_id": user_id.upper(),
        "date": str(top_record['date']),
        "risk_score": float(top_record['risk_score']),
        "risk_level": top_record['risk_level'],
        "feature_attributions": explanation,
        "summary": _build_explanation_summary(user_id.upper(), top_record, sorted_attrs, feature_labels)
    }


def _build_explanation_summary(user_id, record, sorted_attrs, labels):
    """Generates a natural language explanation of the risk score."""
    top_3 = [(labels.get(f, f), p) for f, p in sorted_attrs[:3] if p > 0.5]
    if not top_3:
        return f"{user_id} shows low risk with no significant behavioral anomalies."

    parts = [f"{name} (+{pts:.1f} pts)" for name, pts in top_3]
    return (
        f"{user_id}'s risk score of {record['risk_score']} was primarily driven by: "
        + ", ".join(parts[:-1])
        + (f", and {parts[-1]}" if len(parts) > 1 else parts[0])
        + "."
    )


class EscalationRequest(BaseModel):
    user_id: str
    action: str  # e.g., "LOCK_ACCOUNT", "REVOKE_USB_ACCESS", "REQUIRE_MFA"

@app.post("/api/v1/mitigate")
def trigger_mitigation(request: EscalationRequest):
    """Validate a requested action without executing an access control."""
    allowed_actions = {"LOCK_ACCOUNT", "REVOKE_USB_ACCESS", "REQUIRE_MFA"}
    action = request.action.upper()
    if action not in allowed_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mitigation action. Choose one of: {', '.join(sorted(allowed_actions))}.",
        )

    return {
        "status": "simulation",
        "user_id": request.user_id.upper(),
        "action_requested": action,
        "action_executed": False,
        "message": "No access control was changed. This endpoint records a validated simulation request only."
    }