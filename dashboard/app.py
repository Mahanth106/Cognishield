# dashboard/app.py
"""
CogniShield SOC Analyst Dashboard with Explainable AI Visualizations.
High-Tech Cyberpunk Dark Theme & Custom Cognitive Neural Sentinel Emblem.
Built with Streamlit + Plotly. Reads from SQLite (preferred) or CSV (fallback).
"""
import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "cognishield_icon.png")
if not os.path.exists(FAVICON_PATH):
    FAVICON_PATH = "dashboard/cognishield_icon.png"

# Page Configuration
st.set_page_config(
    page_title="CogniShield | Autonomous Threat Sentinel",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SVG Cognitive Neural Sentinel Logo (single-line string to prevent markdown pre/code block parsing)
COGNISHIELD_LOGO_SVG = (
    '<svg width="58" height="58" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 10px rgba(6,182,212,0.5));">'
    '<defs>'
    '<linearGradient id="shieldGrad" x1="10" y1="5" x2="90" y2="95" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#00f5ff"/><stop offset="50%" stop-color="#6366f1"/><stop offset="100%" stop-color="#d946ef"/>'
    '</linearGradient>'
    '<linearGradient id="coreGrad" x1="38" y1="38" x2="62" y2="62" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#38bdf8"/><stop offset="100%" stop-color="#818cf8"/>'
    '</linearGradient>'
    '<radialGradient id="pulseGlow" cx="50%" cy="50%" r="50%">'
    '<stop offset="0%" stop-color="#00f5ff" stop-opacity="0.35"/><stop offset="100%" stop-color="#00f5ff" stop-opacity="0"/>'
    '</radialGradient>'
    '</defs>'
    '<circle cx="50" cy="50" r="44" fill="url(#pulseGlow)"/>'
    '<polygon points="50,6 88,22 88,58 50,94 12,58 12,22" stroke="url(#shieldGrad)" stroke-width="2.8" fill="#070c17" fill-opacity="0.95"/>'
    '<polygon points="50,14 80,27 80,54 50,84 20,54 20,27" stroke="rgba(99,102,241,0.35)" stroke-width="1.2" stroke-dasharray="3 2" fill="none"/>'
    '<path d="M 32 30 A 24 24 0 0 1 68 30" stroke="#00f5ff" stroke-width="1.5" stroke-linecap="round" opacity="0.85"/>'
    '<path d="M 28 65 A 28 28 0 0 0 72 65" stroke="#d946ef" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>'
    '<line x1="50" y1="27" x2="50" y2="50" stroke="#00f5ff" stroke-width="1.6" opacity="0.8"/>'
    '<line x1="32" y1="41" x2="50" y2="50" stroke="#38bdf8" stroke-width="1.6" opacity="0.8"/>'
    '<line x1="68" y1="41" x2="50" y2="50" stroke="#38bdf8" stroke-width="1.6" opacity="0.8"/>'
    '<line x1="35" y1="62" x2="50" y2="50" stroke="#c084fc" stroke-width="1.6" opacity="0.8"/>'
    '<line x1="65" y1="62" x2="50" y2="50" stroke="#c084fc" stroke-width="1.6" opacity="0.8"/>'
    '<line x1="50" y1="50" x2="50" y2="73" stroke="#e879f9" stroke-width="1.6" opacity="0.8"/>'
    '<circle cx="50" cy="27" r="3.2" fill="#00f5ff"/>'
    '<circle cx="32" cy="41" r="2.8" fill="#38bdf8"/>'
    '<circle cx="68" cy="41" r="2.8" fill="#38bdf8"/>'
    '<circle cx="35" cy="62" r="2.8" fill="#c084fc"/>'
    '<circle cx="65" cy="62" r="2.8" fill="#c084fc"/>'
    '<circle cx="50" cy="73" r="2.8" fill="#e879f9"/>'
    '<polygon points="50,42 58,50 50,58 42,50" fill="url(#coreGrad)"/>'
    '<circle cx="50" cy="50" r="2.2" fill="#ffffff"/>'
    '</svg>'
)

# Premium Cybersecurity Dark SOC Theme CSS
st.markdown("""
<style>
/* Base page layout */
.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 96%;
}

/* Header Container */
.header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: linear-gradient(135deg, rgba(14, 21, 36, 0.95) 0%, rgba(8, 12, 20, 0.95) 100%);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    margin-bottom: 20px;
}

.header-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.header-title-box h1 {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 0;
    padding: 0;
    background: linear-gradient(135deg, #00f5ff 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
}

.header-title-box p {
    margin: 3px 0 0 0;
    font-size: 13px;
    color: #94a3b8;
    letter-spacing: 0.4px;
    font-weight: 500;
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 10px;
}

.soc-badge {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: #cbd5e1;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-family: monospace;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.badge-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}

/* Metric Cards Styling */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0e1524 0%, #0a0f1a 100%) !important;
    border: 1px solid rgba(30, 41, 59, 0.8) !important;
    border-top: 2px solid rgba(6, 182, 212, 0.6) !important;
    padding: 14px 18px !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
}

div[data-testid="stMetricLabel"] p {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}

div[data-testid="stMetricValue"] div {
    color: #f8fafc !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* Incident Feed Threat Card */
.threat-card {
    background: linear-gradient(135deg, rgba(26, 15, 20, 0.85) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 4px solid #ef4444;
    padding: 14px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.12);
}

.threat-card h4 {
    margin: 0;
    color: #f87171;
    font-size: 14.5px;
    font-weight: 700;
}

.threat-card p {
    margin: 4px 0 0 0;
    color: #cbd5e1;
    font-size: 12.5px;
}

/* SHAP AI Explanation Card */
.shap-card {
    background: linear-gradient(145deg, rgba(19, 16, 36, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-left: 4px solid #a855f7;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.15);
    margin-bottom: 14px;
}

.section-header {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "cognishield.db")
FEED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "threat_intelligence_feed.csv")


@st.cache_data(ttl=5)
def load_data():
    """Load threat intelligence from SQLite (preferred) or CSV (fallback)."""
    df = None

    # Prefer SQLite
    if os.path.exists(DB_PATH):
        try:
            from src.database import query_all_threats
            df = query_all_threats(DB_PATH)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception:
            pass

    # Fallback to CSV
    if os.path.exists(FEED_PATH):
        df = pd.read_csv(FEED_PATH)
        df['date'] = pd.to_datetime(df['date'])
        return df

    return None


# ── FEATURE LABELS (human-readable) ──────────────────────────────
FEATURE_LABELS = {
    'total_logons': 'Total Logons',
    'total_logoffs': 'Total Logoffs',
    'after_hours_logons': 'After-Hours Logons',
    'weekend_logons': 'Weekend Logons',
    'usb_connects': 'USB Connections',
    'files_accessed': 'Files Accessed',
    'files_copied_external': 'External File Copies',
    'lstm_temporal': 'Temporal Sequence Drift',
    'nlp_lexical': 'Lexical Threat Keywords'
}

df = load_data()

# Custom Header with Vector Cognitive Logo (Single-line construction to prevent pre/code blocks)
header_html = (
    '<div class="header-container">'
    '<div class="header-brand">'
    f'{COGNISHIELD_LOGO_SVG}'
    '<div class="header-title-box">'
    '<h1>COGNISHIELD</h1>'
    '<p>Context-Aware Insider Threat Sentinel &bull; Multi-Modal AI Telemetry &bull; Zero-Trust Audit</p>'
    '</div>'
    '</div>'
    '<div class="header-badges">'
    '<div class="soc-badge"><span class="badge-pulse"></span> TELEMETRY ACTIVE</div>'
    '<div class="soc-badge">ENGINE: HYBRID AI / SHAP</div>'
    '<div class="soc-badge">STORAGE: SQLITE</div>'
    '</div>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

if df is None or df.empty:
    st.error("Threat Intelligence Feed missing. Run `python run_pipeline.py` first to ingest logs.")
    st.stop()

# Top KPI Metric Cards
high_count = len(df[df['risk_level'] == 'High'])
med_count = len(df[df['risk_level'] == 'Medium'])
low_count = len(df[df['risk_level'] == 'Low'])
total_entities = df['user'].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monitored Entities", total_entities)
c2.metric("High-Risk Alerts", high_count, delta="Action Required" if high_count > 0 else "Nominal", delta_color="inverse")
c3.metric("Medium-Risk Anomalies", med_count)
c4.metric("Low-Risk Baselines", low_count)

st.write("")

# Main Layout Split
col_left, col_right = st.columns([1.1, 1.3])

with col_left:
    st.markdown('<div class="section-header">🚨 Priority Incident Feed</div>', unsafe_allow_html=True)
    high_threats = df[df['risk_level'] == 'High'].sort_values(by=['risk_score', 'date'], ascending=False)

    if not high_threats.empty:
        for _, row in high_threats.iterrows():
            st.markdown(f"""<div class="threat-card">
<h4>⚡ CRITICAL ESCALATION &bull; {row['user']}</h4>
<p><b>Observation:</b> {row['date'].strftime('%Y-%m-%d')} &nbsp;|&nbsp; <b>Risk Score:</b> <span style="color:#ef4444; font-weight:800;">{row['risk_score']} / 100</span></p>
<p style="opacity:0.85; font-size:12px;">📂 {int(row['files_copied_external'])} External Files &bull; 🔌 {int(row['usb_connects'])} USB Events &bull; 🌙 {int(row['after_hours_logons'])} Off-Hours &bull; 🚩 {int(row['flagged_terms'])} Red-Flag Phrases</p>
</div>""", unsafe_allow_html=True)
    else:
        st.success("No active critical threats detected across current fleet telemetry.")

    st.write("")
    st.markdown('<div class="section-header">📊 Fleet Threat Posture</div>', unsafe_allow_html=True)

    # Fleet Threat Posture
    risk_counts = df['risk_level'].value_counts()
    total_events = len(df)

    low_cnt = risk_counts.get('Low', 0)
    med_cnt = risk_counts.get('Medium', 0)
    high_cnt = risk_counts.get('High', 0)

    low_pct = round((low_cnt / total_events) * 100, 2)
    med_pct = round((med_cnt / total_events) * 100, 2)
    high_pct = round((high_cnt / total_events) * 100, 2)

    posture_df = pd.DataFrame({
        'Threat Level': ['High Risk', 'Medium Risk', 'Low Risk'],
        'Count': [high_cnt, med_cnt, low_cnt],
        'Percentage': [high_pct, med_pct, low_pct],
        'Label': [f"{high_cnt} ({high_pct}%)", f"{med_cnt} ({med_pct}%)", f"{low_cnt} ({low_pct}%)"]
    })

    fig_posture = px.bar(
        posture_df,
        x='Percentage',
        y='Threat Level',
        orientation='h',
        text='Label',
        color='Threat Level',
        color_discrete_map={
            'High Risk': '#ef4444',
            'Medium Risk': '#f59e0b',
            'Low Risk': '#10b981'
        }
    )

    fig_posture.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(color='#cbd5e1', size=11)
    )

    fig_posture.update_layout(
        paper_bgcolor='rgba(14, 21, 36, 0.65)',
        plot_bgcolor='rgba(8, 12, 20, 0.75)',
        font=dict(color='#cbd5e1', family="sans-serif"),
        height=210,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="Telemetry Share (%)", range=[0, 115], gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)'),
        yaxis=dict(title="", gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)')
    )

    st.plotly_chart(fig_posture, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">🔍 Entity Forensic Behavioral Audit</div>', unsafe_allow_html=True)
    user_list = sorted(df['user'].unique())
    # Dynamic audit targeting: automatically default to the highest-risk entity identified by the AI engine
    top_threat_entity = df.sort_values(by='risk_score', ascending=False)['user'].iloc[0]
    default_idx = user_list.index(top_threat_entity) if top_threat_entity in user_list else 0
    selected_user = st.selectbox(
        "Select Monitored Entity to Inspect:",
        options=user_list,
        index=default_idx,
        help="Defaults automatically to the highest-risk entity identified by the AI engine."
    )

    user_data = df[df['user'] == selected_user].sort_values(by='date')

    # Trajectory Timeline
    fig_timeline = px.line(
        user_data,
        x='date',
        y='risk_score',
        markers=True,
        title=f"Longitudinal Risk Trajectory // {selected_user}",
        labels={'risk_score': 'Composite Risk Score (0-100)', 'date': 'Timeline'},
        range_y=[0, 105]
    )
    fig_timeline.update_traces(
        line=dict(color='#00f5ff', width=2.8),
        marker=dict(size=7, color='#38bdf8', line=dict(color='#0e1524', width=1.5))
    )
    fig_timeline.add_hline(y=65, line_dash="dash", line_color="#ef4444", annotation_text="High Threat (65)", annotation_font_color="#f87171")
    fig_timeline.add_hline(y=35, line_dash="dash", line_color="#f59e0b", annotation_text="Medium Threat (35)", annotation_font_color="#fbbf24")
    
    fig_timeline.update_layout(
        paper_bgcolor='rgba(14, 21, 36, 0.65)',
        plot_bgcolor='rgba(8, 12, 20, 0.75)',
        font=dict(color='#cbd5e1', family="sans-serif"),
        height=270,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)'),
        yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)')
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Multi-Model Subsystem Severity
    latest_rec = user_data.iloc[-1]
    st.markdown(f"<span style='color:#94a3b8; font-size:13px; font-weight:600;'>MULTI-MODEL TELEMETRY DECOMPOSITION ({latest_rec['date'].strftime('%Y-%m-%d')}):</span>", unsafe_allow_html=True)

    model_metrics = pd.DataFrame({
        "Model Subsystem": ["Deep Autoencoder", "Temporal LSTM", "Isolation Forest", "NLP Lexical Scanner"],
        "Severity": [latest_rec['ae_score'], latest_rec['lstm_score'], latest_rec['if_score'], latest_rec['lexical_risk_score']]
    })

    fig_bar = px.bar(
        model_metrics,
        x="Severity",
        y="Model Subsystem",
        orientation='h',
        color="Severity",
        color_continuous_scale=[[0, '#0284c7'], [0.5, '#f59e0b'], [1, '#ef4444']],
        range_x=[0, 1.0]
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(14, 21, 36, 0.65)',
        plot_bgcolor='rgba(8, 12, 20, 0.75)',
        font=dict(color='#cbd5e1', family="sans-serif"),
        height=190,
        margin=dict(t=15, b=15, l=10, r=10),
        xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)'),
        yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)')
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── SHAP Explainable AI Section ──────────────────────────────────
st.write("---")
st.markdown('<div class="section-header">🧠 Explainable AI: Feature Attribution Decomposition</div>', unsafe_allow_html=True)

has_shap = 'shap_attributions' in df.columns and user_data['shap_attributions'].notna().any()

if has_shap:
    user_sorted = user_data.sort_values(by='risk_score', ascending=False)
    top_record = user_sorted.iloc[0]

    shap_col1, shap_col2 = st.columns([1.3, 1])

    with shap_col1:
        try:
            attributions = json.loads(top_record['shap_attributions'])

            attr_data = []
            for feat, pts in attributions.items():
                attr_data.append({
                    'Feature': FEATURE_LABELS.get(feat, feat),
                    'Contribution (pts)': pts,
                    'Color': '#ef4444' if pts >= 8 else ('#f59e0b' if pts >= 3 else '#10b981')
                })

            attr_df = pd.DataFrame(attr_data).sort_values(by='Contribution (pts)', ascending=True)

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=attr_df['Feature'],
                x=attr_df['Contribution (pts)'],
                orientation='h',
                marker=dict(
                    color=[
                        '#ef4444' if v >= 8 else ('#f59e0b' if v >= 3 else '#10b981')
                        for v in attr_df['Contribution (pts)']
                    ],
                    line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
                ),
                text=[f"+{v:.1f}" if v >= 0.5 else f"{v:.1f}" for v in attr_df['Contribution (pts)']],
                textposition='outside',
                textfont=dict(color='#cbd5e1', size=11),
                hovertemplate='<b>%{y}</b>: +%{x:.2f} risk points<extra></extra>'
            ))

            fig_shap.update_layout(
                paper_bgcolor='rgba(14, 21, 36, 0.65)',
                plot_bgcolor='rgba(8, 12, 20, 0.75)',
                font=dict(color='#cbd5e1', family="sans-serif"),
                title=f"SHAP Attribution Breakdown &bull; {selected_user} ({top_record['date'].strftime('%Y-%m-%d')})",
                xaxis=dict(title="Risk Score Contribution (points)", gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)'),
                yaxis=dict(title="", gridcolor='rgba(148, 163, 184, 0.1)', zerolinecolor='rgba(148, 163, 184, 0.15)'),
                height=350,
                showlegend=False,
                margin=dict(t=40, b=20, l=10, r=60)
            )

            st.plotly_chart(fig_shap, use_container_width=True)

        except (json.JSONDecodeError, KeyError):
            st.warning("Unable to decode SHAP attributions for this entry.")

    with shap_col2:
        try:
            attributions = json.loads(top_record['shap_attributions'])
            sorted_attrs = sorted(attributions.items(), key=lambda x: x[1], reverse=True)

            top_3 = [(FEATURE_LABELS.get(f, f), p) for f, p in sorted_attrs[:3] if p > 0.5]

            if top_3:
                summary_parts = [f"<b style='color:#00f5ff;'>{name}</b> (+{pts:.1f} pts)" for name, pts in top_3]
                summary_text = ", ".join(summary_parts[:-1]) + (f", and {summary_parts[-1]}" if len(summary_parts) > 1 else summary_parts[0])
                risk_color = "#ef4444" if top_record['risk_score'] >= 65 else ("#f59e0b" if top_record['risk_score'] >= 35 else "#10b981")

                st.markdown(f"""<div class="shap-card">
<h4 style="margin:0 0 10px 0; color:#c084fc; font-size:16px;">🧠 AI Contextual Attribution</h4>
<p style="margin:4px 0; font-size:13.5px; line-height:1.5;">Entity <b style="color:#f8fafc;">{selected_user}</b> reached a risk score of <b style="color:{risk_color};">{top_record['risk_score']}</b> on {top_record['date'].strftime('%Y-%m-%d')}.</p>
<p style="margin:8px 0; font-size:13px; color:#cbd5e1; line-height:1.6;">Primary anomaly drivers identified by the ensemble: {summary_text}.</p>
<div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.08); font-size:11.5px; color:#94a3b8;">Total Explainable Points: <b>{sum(v for _, v in sorted_attrs):.1f}</b> &bull; Attribution method: <b>SHAP + reconstruction error</b></div>
</div>""", unsafe_allow_html=True)
            else:
                st.info(f"{selected_user} demonstrates normal baseline behavioral patterns.")

            st.markdown("<span style='color:#94a3b8; font-size:12.5px; font-weight:600;'>SIGNAL CONTRIBUTIONS TABLE:</span>", unsafe_allow_html=True)
            detail_data = []
            for feat, pts in sorted_attrs:
                detail_data.append({
                    'Signal': FEATURE_LABELS.get(feat, feat),
                    'Impact (Pts)': round(pts, 2),
                    'Level': '🔴 Critical' if pts >= 8 else ('🟡 Moderate' if pts >= 3 else '🟢 Normal')
                })
            st.dataframe(
                pd.DataFrame(detail_data),
                hide_index=True,
                use_container_width=True,
                height=230
            )

        except (json.JSONDecodeError, KeyError):
            st.warning("Unable to parse SHAP attributions.")
else:
    st.info("SHAP explainability telemetry not available. Re-run `python run_pipeline.py`.")


# Contextual Log Ledger Section
st.write("---")
st.markdown('<div class="section-header">📋 Contextual Telemetry Ledger</div>', unsafe_allow_html=True)
filter_level = st.multiselect("Filter Severity Category:", options=["High", "Medium", "Low"], default=["High", "Medium"])
filtered_df = df[df['risk_level'].isin(filter_level)].sort_values(by='risk_score', ascending=False).copy()

filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')

display_columns = {
    'date': 'Date',
    'user': 'Entity ID',
    'risk_score': 'Risk Rating',
    'risk_level': 'Threat Class',
    'after_hours_logons': 'Off-Hours',
    'usb_connects': 'USB Connects',
    'files_accessed': 'Files Read',
    'files_copied_external': 'External Copies',
    'lexical_risk_score': 'NLP Threat',
    'flagged_terms': 'Red Flags'
}

st.dataframe(
    filtered_df[list(display_columns.keys())].rename(columns=display_columns),
    hide_index=True,
    use_container_width=True,
    height=270
)