# src/risk_engine.py
"""
Dynamic Multi-Subsystem Risk Scoring Engine with Explainable AI Attribution.

Combines Autoencoder reconstruction anomaly, LSTM temporal deviation,
Isolation Forest boundary anomaly, and NLP lexical risk into a unified
0-100 composite risk score with per-feature SHAP-based attributions.
"""
import json
import numpy as np
import pandas as pd


class CogniShieldRiskEngine:
    def __init__(self, w_ae=0.35, w_lstm=0.25, w_if=0.20, w_nlp=0.20):
        self.w_ae = w_ae
        self.w_lstm = w_lstm
        self.w_if = w_if
        self.w_nlp = w_nlp

    def compute_risk_profiles(self, df: pd.DataFrame, ae_scores: np.ndarray,
                              lstm_scores: np.ndarray, if_scores: np.ndarray,
                              shap_attributions=None) -> pd.DataFrame:
        """
        Combines model scores with textual/NLP indicators into a unified 0-100 risk rating.
        Optionally includes SHAP feature attribution JSON strings.
        """
        nlp_scores = df['lexical_risk_score'].to_numpy()

        composite_score = (
            (self.w_ae * ae_scores) +
            (self.w_lstm * lstm_scores) +
            (self.w_if * if_scores) +
            (self.w_nlp * nlp_scores)
        ) * 100.0

        risk_scores = np.round(np.clip(composite_score, 0, 100), 2)

        # Categorical risk assignment
        levels = []
        for r in risk_scores:
            if r >= 65.0:
                levels.append("High")
            elif r >= 35.0:
                levels.append("Medium")
            else:
                levels.append("Low")

        result_df = df.copy()
        result_df['ae_score'] = np.round(ae_scores, 4)
        result_df['lstm_score'] = np.round(lstm_scores, 4)
        result_df['if_score'] = np.round(if_scores, 4)
        result_df['risk_score'] = risk_scores
        result_df['risk_level'] = levels

        # Attach SHAP attributions if provided
        if shap_attributions is not None:
            result_df['shap_attributions'] = shap_attributions

        return result_df

    def compute_feature_attributions(self, shap_values: np.ndarray, ae_deltas: np.ndarray,
                                     ae_scores: np.ndarray, if_scores: np.ndarray,
                                     lstm_scores: np.ndarray, nlp_scores: np.ndarray,
                                     feature_names: list) -> list:
        """
        Combines IF SHAP values and AE reconstruction deltas into unified
        per-feature risk attributions. The sum of all attributions ≈ risk_score.

        For each sample, each behavioral feature receives a point contribution from:
          - Isolation Forest: proportional to its SHAP magnitude
          - Autoencoder: proportional to its reconstruction error delta
        
        Non-decomposable contributions (LSTM temporal, NLP lexical) are added
        as aggregate entries.

        Args:
            shap_values:   (n_samples, n_features) SHAP values from Isolation Forest
            ae_deltas:     (n_samples, n_features) per-feature AE reconstruction errors
            ae_scores:     (n_samples,) normalized AE anomaly scores
            if_scores:     (n_samples,) normalized IF anomaly scores
            lstm_scores:   (n_samples,) normalized LSTM sequence deviation scores
            nlp_scores:    (n_samples,) lexical risk scores
            feature_names: list of feature column names

        Returns:
            list of JSON strings, one per sample, with per-feature point contributions
        """
        n_samples = len(ae_scores)
        attributions = []

        for i in range(n_samples):
            attr = {}

            # IF contribution: distribute IF's risk points across features by SHAP proportion
            shap_row = np.abs(shap_values[i])
            shap_total = shap_row.sum()
            shap_proportions = (shap_row / shap_total) if shap_total > 0 else np.zeros_like(shap_row)

            # AE contribution: distribute AE's risk points across features by delta proportion
            ae_row = ae_deltas[i]
            ae_total = ae_row.sum()
            ae_proportions = (ae_row / ae_total) if ae_total > 0 else np.zeros_like(ae_row)

            # Total risk points from each model subsystem
            if_risk_pts = float(if_scores[i] * self.w_if * 100)
            ae_risk_pts = float(ae_scores[i] * self.w_ae * 100)

            for j, name in enumerate(feature_names):
                contribution = (shap_proportions[j] * if_risk_pts) + (ae_proportions[j] * ae_risk_pts)
                attr[name] = round(float(contribution), 2)

            # Non-decomposable model contributions (aggregated)
            attr['lstm_temporal'] = round(float(lstm_scores[i] * self.w_lstm * 100), 2)
            attr['nlp_lexical'] = round(float(nlp_scores[i] * self.w_nlp * 100), 2)

            attributions.append(json.dumps(attr))

        return attributions