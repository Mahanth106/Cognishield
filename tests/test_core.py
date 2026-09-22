import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from src.api import EscalationRequest, require_api_token, trigger_mitigation
from src.database import init_database, query_all_features, query_all_threats, store_pipeline_results


class CoreContractTests(unittest.TestCase):
    def test_mitigation_payload_is_strict_and_simulated(self):
        request = EscalationRequest(user_id=" user_004 ", action="lock_account")
        response = trigger_mitigation(request)
        self.assertEqual(response["user_id"], "USER_004")
        self.assertEqual(response["action_requested"], "LOCK_ACCOUNT")
        self.assertFalse(response["action_executed"])

        with self.assertRaises(ValidationError):
            EscalationRequest(user_id="anything", action="DELETE_EVERYTHING")

    def test_development_auth_allows_local_requests(self):
        original_environment = os.environ.get("COGNISHIELD_ENV")
        try:
            os.environ["COGNISHIELD_ENV"] = "development"
            require_api_token(None)
        finally:
            if original_environment is None:
                os.environ.pop("COGNISHIELD_ENV", None)
            else:
                os.environ["COGNISHIELD_ENV"] = original_environment

    def test_atomic_pipeline_storage_replaces_both_tables(self):
        feature_row = {
            "date": "2026-08-12", "user": "USER_004", "total_logons": 1,
            "total_logoffs": 1, "after_hours_logons": 1, "weekend_logons": 0,
            "usb_connects": 4, "files_accessed": 64, "files_copied_external": 51,
            "lexical_risk_score": 1, "flagged_terms": 3,
        }
        threat_row = {
            **feature_row, "ae_score": 0.9, "lstm_score": 1.0, "if_score": 0.8,
            "risk_score": 82.13, "risk_level": "High", "shap_attributions": "{}",
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db_path = str(Path(directory) / "test.db")
            init_database(db_path)
            store_pipeline_results(pd.DataFrame([feature_row]), pd.DataFrame([threat_row]), db_path)
            self.assertEqual(len(query_all_features(db_path)), 1)
            self.assertEqual(len(query_all_threats(db_path)), 1)


if __name__ == "__main__":
    unittest.main()
