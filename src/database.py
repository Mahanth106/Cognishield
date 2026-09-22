# src/database.py
"""
SQLite Persistence Layer for CogniShield Threat Intelligence.
Uses Python's built-in sqlite3 module — zero external dependencies.
Replaces CSV-based persistence with structured SQL queries while
maintaining CSV export as a fallback.
"""
import os
import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "cognishield.db")


def init_database(db_path=DEFAULT_DB_PATH):
    """
    Initialize SQLite database with schema for features and threat intelligence.
    Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Daily behavioral feature vectors (output of feature_extraction.py)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_daily_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            user TEXT NOT NULL,
            total_logons REAL DEFAULT 0,
            total_logoffs REAL DEFAULT 0,
            after_hours_logons REAL DEFAULT 0,
            weekend_logons REAL DEFAULT 0,
            usb_connects REAL DEFAULT 0,
            files_accessed REAL DEFAULT 0,
            files_copied_external REAL DEFAULT 0,
            lexical_risk_score REAL DEFAULT 0,
            flagged_terms REAL DEFAULT 0,
            UNIQUE(date, user)
        )
    """)

    # Scored threat intelligence feed (output of risk_engine.py)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threat_intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            user TEXT NOT NULL,
            total_logons REAL DEFAULT 0,
            total_logoffs REAL DEFAULT 0,
            after_hours_logons REAL DEFAULT 0,
            weekend_logons REAL DEFAULT 0,
            usb_connects REAL DEFAULT 0,
            files_accessed REAL DEFAULT 0,
            files_copied_external REAL DEFAULT 0,
            lexical_risk_score REAL DEFAULT 0,
            flagged_terms REAL DEFAULT 0,
            ae_score REAL DEFAULT 0,
            lstm_score REAL DEFAULT 0,
            if_score REAL DEFAULT 0,
            risk_score REAL DEFAULT 0,
            risk_level TEXT DEFAULT 'Low',
            shap_attributions TEXT,
            UNIQUE(date, user)
        )
    """)

    # Indexes for frequent query patterns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ti_user ON threat_intelligence(user)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ti_risk ON threat_intelligence(risk_level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ti_score ON threat_intelligence(risk_score DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_user ON user_daily_features(user)")

    conn.commit()
    conn.close()
    return db_path


def store_features(df, db_path=DEFAULT_DB_PATH):
    """Bulk insert/replace the feature matrix into SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM user_daily_features")
    df.to_sql("user_daily_features", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def store_threat_feed(df, db_path=DEFAULT_DB_PATH):
    """Bulk insert/replace the scored threat intelligence feed into SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM threat_intelligence")
    df.to_sql("threat_intelligence", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


# ── Query Functions ──────────────────────────────────────────────

def query_all_threats(db_path=DEFAULT_DB_PATH):
    """Retrieve complete threat intelligence feed as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM threat_intelligence ORDER BY risk_score DESC", conn
    )
    conn.close()
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df


def query_high_risk(db_path=DEFAULT_DB_PATH):
    """Retrieve only High-risk alerts, sorted by severity."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM threat_intelligence WHERE risk_level = 'High' ORDER BY risk_score DESC",
        conn,
    )
    conn.close()
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df


def query_user_history(user_id, db_path=DEFAULT_DB_PATH):
    """Retrieve longitudinal risk history for a specific user entity."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM threat_intelligence WHERE user = ? ORDER BY date",
        conn,
        params=(user_id.upper(),),
    )
    conn.close()
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df


def query_all_features(db_path=DEFAULT_DB_PATH):
    """Retrieve complete feature matrix as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM user_daily_features ORDER BY user, date", conn
    )
    conn.close()
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df


def query_user_features(user_id, db_path=DEFAULT_DB_PATH):
    """Retrieve feature data for a specific user entity."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM user_daily_features WHERE user = ? ORDER BY date",
        conn,
        params=(user_id.upper(),),
    )
    conn.close()
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df
