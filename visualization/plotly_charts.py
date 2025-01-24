# visualization/plotly_charts.py

import plotly.express as px
import plotly.graph_objects as go

def plot_risk_scores(df):
    """
    Plot risk scores using a bar chart.
    """
    return px.bar(df, x=df.index, y="Risk Score", title="Risk Scores per Entry")

def plot_anomaly_distribution(df):
    """
    Plot anomaly distribution using a pie chart.
    """
    anomaly_counts = df["Anomaly"].value_counts()
    return px.pie(anomaly_counts, values=anomaly_counts.values, names=anomaly_counts.index, title="Anomaly Distribution")

def plot_scatter_anomalies(df, y_column):
    """
    Plot anomalies using a scatter plot.
    """
    return px.scatter(df, x=df.index, y=y_column, color="Anomaly", title="Anomalies in Risk Scores")

def plot_heatmap(df, x_column, y_column):
    """
    Plot a heatmap for anomalies.
    """
    return px.density_heatmap(df, x=x_column, y=y_column, facet_col="Anomaly", title="Anomaly Heatmap")

def plot_histogram(df, column, title):
    """
    Plot a histogram for a specific column.
    """
    return px.histogram(df, x=column, title=title)

def plot_bar_chart(df, x, y, title):
    """
    Create a bar chart using Plotly.
    """
    fig = px.bar(df, x=x, y=y, title=title)
    return fig

def plot_pie_chart(data, title):
    """
    Create a pie chart using Plotly.
    """
    fig = px.pie(data, values=data.values, names=data.index, title=title)
    return fig