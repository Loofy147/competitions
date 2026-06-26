import pandas as pd
from typing import Dict, Any

class MonitoringSkill:
    """
    Skill: Competition Monitoring.
    Tracks leaderboard status and submission history.
    """
    def __init__(self, client):
        self.client = client

    def get_status(self, competition_id: str) -> Dict[str, Any]:
        """Returns the current status of the competition for the user."""
        submissions_raw = self.client.list_submissions(competition_id)
        # In practice, parse the raw CLI output into a structured dict
        return {"raw_status": submissions_raw}

    def log_improvement(self, prev_score: float, new_score: float, metric_name: str):
        """Logs architectural improvements."""
        gain = new_score - prev_score
        print(f"[*] Improvement Detected: {metric_name} changed by {gain:+.4f} ({prev_score:.4f} -> {new_score:.4f})")
