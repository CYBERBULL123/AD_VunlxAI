# ids/intrusion_detection.py

import pandas as pd

class IntrusionDetectionSystem:
    def __init__(self):
        self.rules = [
            {"rule": "Port Scan", "condition": lambda x: x["Port"] == 22 and x["State"] == "open"},
            {"rule": "SQL Injection", "condition": lambda x: "sql" in x["Service"].lower()},
            {"rule": "Brute Force", "condition": lambda x: x["Port"] == 21 and x["State"] == "open"},
        ]

    def detect_intrusions(self):
        """
        Detect intrusions based on predefined rules.
        """
        # Simulate network traffic data
        traffic_data = [
            {"Port": 22, "State": "open", "Service": "ssh"},
            {"Port": 80, "State": "open", "Service": "http"},
            {"Port": 21, "State": "open", "Service": "ftp"},
        ]
        alerts = []
        for data in traffic_data:
            for rule in self.rules:
                if rule["condition"](data):
                    alerts.append({"Rule": rule["rule"], "Details": data})
        return pd.DataFrame(alerts)