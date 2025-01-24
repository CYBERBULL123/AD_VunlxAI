# models/anomaly_detection.py

from sklearn.ensemble import IsolationForest
from config import CONTAMINATION

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=CONTAMINATION, random_state=42)

    def detect_anomalies(self, data):
        """
        Detect anomalies in the data.
        """
        return self.model.fit_predict(data)