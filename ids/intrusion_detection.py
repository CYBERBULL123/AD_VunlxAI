# ids/intrusion_detection.py

import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IntrusionDetectionSystem:
    def __init__(self):
        """
        Initialize the IDS with predefined rules and a machine learning model (optional).
        """
        self.rules = [
            {"rule": "Port Scan", "condition": lambda x: x["Port"] == 22 and x["State"] == "open"},
            {"rule": "SQL Injection", "condition": lambda x: "sql" in x["Service"].lower() and "drop table" in x["Payload"].lower()},
            {"rule": "Brute Force", "condition": lambda x: x["Port"] == 21 and x["State"] == "open" and x["Packet Count"] > 50},
            {"rule": "DDoS Attack", "condition": lambda x: x["Packet Count"] > 1000},  # Example threshold
            {"rule": "XSS Attack", "condition": lambda x: "<script>" in x["Payload"]},  # Example payload check
            {"rule": "Malware Download", "condition": lambda x: "exe" in x["Payload"].lower() or "dll" in x["Payload"].lower()},
            {"rule": "Phishing Attempt", "condition": lambda x: "login" in x["Payload"].lower() and "http://fake.com" in x["Payload"].lower()},
            {"rule": "Unusual Traffic", "condition": lambda x: x["Packet Count"] > 500 and x["Port"] not in [80, 443, 22]},
        ]
        self.alerts = []  # Store detected alerts
        self.traffic_data = []  # Simulate real-time traffic data

    def detect_intrusions(self, traffic_data):
        """
        Detect intrusions based on predefined rules and machine learning (if implemented).
        """
        alerts = []
        for data in traffic_data:
            for rule in self.rules:
                try:
                    if rule["condition"](data):
                        alert = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Rule": rule["rule"],
                            "Details": data,
                        }
                        alerts.append(alert)
                        logger.warning(f"Intrusion Detected: {alert}")
                except KeyError as e:
                    logger.error(f"Missing key in traffic data: {e}")
        return pd.DataFrame(alerts)

    def simulate_traffic(self):
        """
        Simulate realistic network traffic for testing purposes.
        """
        self.traffic_data = [
            {"Port": 22, "State": "open", "Service": "ssh", "Packet Count": 10, "Payload": ""},
            {"Port": 80, "State": "open", "Service": "http", "Packet Count": 500, "Payload": "<script>alert('XSS')</script>"},
            {"Port": 21, "State": "open", "Service": "ftp", "Packet Count": 2000, "Payload": ""},
            {"Port": 3306, "State": "open", "Service": "mysql", "Packet Count": 50, "Payload": "SELECT * FROM users; DROP TABLE users;"},
            {"Port": 443, "State": "open", "Service": "https", "Packet Count": 100, "Payload": "Download malware.exe"},
            {"Port": 8080, "State": "open", "Service": "http", "Packet Count": 1500, "Payload": "http://fake.com/login"},
            {"Port": 53, "State": "open", "Service": "dns", "Packet Count": 200, "Payload": ""},
        ]

    def monitor_traffic(self):
        """
        Simulate real-time monitoring of network traffic.
        """
        self.simulate_traffic()  # Simulate incoming traffic
        alerts = self.detect_intrusions(self.traffic_data)
        if not alerts.empty:
            logger.info("Intrusions detected. Sending alerts...")
            self.send_alerts(alerts)
        else:
            logger.info("No intrusions detected.")

    def send_alerts(self, alerts):
        """
        Send alerts to administrators in a clear and professional format.
        """
        for _, alert in alerts.iterrows():
            logger.warning(
                f"🚨 ALERT: {alert['Rule']} detected at {alert['Timestamp']}.\n"
                f"   Details: Port={alert['Details']['Port']}, Service={alert['Details']['Service']}, "
                f"Packet Count={alert['Details']['Packet Count']}, Payload={alert['Details']['Payload']}"
            )
