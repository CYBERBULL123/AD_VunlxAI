from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import CONTAMINATION

class AnomalyDetector:
    def __init__(self, algorithm="isolation_forest", contamination=CONTAMINATION, n_estimators=100, max_samples="auto", max_features=1.0, novelty=False):
        """
        Initialize the anomaly detector with the specified algorithm and parameters.

        Parameters:
            algorithm (str): The anomaly detection algorithm to use. Options: "isolation_forest", "local_outlier_factor", "one_class_svm".
            contamination (float): The proportion of outliers in the data.
            n_estimators (int): The number of base estimators in the ensemble (for Isolation Forest).
            max_samples (int or float): The number of samples to draw for training each base estimator (for Isolation Forest).
            max_features (int or float): The number of features to draw for training each base estimator (for Isolation Forest).
            novelty (bool): Whether to use the model for novelty detection (for Isolation Forest and One-Class SVM).
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.novelty = novelty

        if self.algorithm == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                random_state=42,
            )
        elif self.algorithm == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=self.novelty,
            )
        elif self.algorithm == "one_class_svm":
            self.model = OneClassSVM(
                nu=self.contamination,  # nu is analogous to contamination
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def detect_anomalies(self, data):
        """
        Detect anomalies in the data.

        Parameters:
            data (array-like): The input data to detect anomalies in.

        Returns:
            anomalies (array): An array of labels (-1 for anomalies, 1 for normal points).
        """
        return self.model.fit_predict(data)

    def evaluate_anomalies(self, data, true_labels):
        """
        Evaluate the performance of the anomaly detection model.

        Parameters:
            data (array-like): The input data.
            true_labels (array-like): The true labels (1 for normal, -1 for anomalies).

        Returns:
            metrics (dict): A dictionary containing precision, recall, and F1-score.
        """
        anomalies = self.detect_anomalies(data)
        precision = precision_score(true_labels, anomalies, pos_label=-1)
        recall = recall_score(true_labels, anomalies, pos_label=-1)
        f1 = f1_score(true_labels, anomalies, pos_label=-1)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def plot_anomalies(self, data, feature_1, feature_2):
        """
        Visualize anomalies in a 2D scatter plot using Plotly.

        Parameters:
            data (array-like): The input data.
            feature_1 (str or int): The first feature to plot.
            feature_2 (str or int): The second feature to plot.

        Returns:
            fig (plotly.graph_objects.Figure): A Plotly figure object.
        """
        anomalies = self.detect_anomalies(data)
        data = np.array(data)

        # Create a DataFrame for Plotly
        import pandas as pd
        df = pd.DataFrame(data, columns=[f"Feature {i}" for i in range(data.shape[1])])
        df["Anomaly"] = ["Anomaly" if a == -1 else "Normal" for a in anomalies]

        # Create the scatter plot
        fig = px.scatter(
            df,
            x=f"Feature {feature_1}",
            y=f"Feature {feature_2}",
            color="Anomaly",
            color_discrete_map={"Normal": "blue", "Anomaly": "red"},
            title="Anomaly Detection",
            labels={"color": "Anomaly Status"},
        )

        return fig

    def feature_importance(self):
        """
        Get feature importance (only applicable for Isolation Forest).

        Returns:
            importance (array): An array of feature importances.
        """
        if self.algorithm != "isolation_forest":
            raise ValueError("Feature importance is only available for Isolation Forest.")

        return self.model.feature_importances_