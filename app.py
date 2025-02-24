import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import time
from pathlib import Path
import plotly.graph_objects as go
import logging
from data_processing.data_cleaning import clean_data, preprocess_data
from data_processing.log_parser import parse_log_file
from models.anomaly_detection import AnomalyDetector
from models.risk_prediction import train_model, predict_vulnerability, predict_exploitation_likelihood , predict_vulnerability_mock , predict_exploitation_likelihood_mock , predict_with_uncertainty , evaluate_model_comprehensive
from network_scanning.nmap_scanner import NmapScanner
from visualization.plotly_charts import (
    plot_risk_scores,
    plot_anomaly_distribution,
    plot_scatter_anomalies,
    plot_heatmap,
    plot_histogram,
    plot_bar_chart,
    plot_pie_chart
)
from ids.intrusion_detection import IntrusionDetectionSystem
from reporting.report_generator import generate_vulnerability_report

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def show_intro():
    """
    Display the intro/overview page.
    """
    st.title("🛡️ Advanced AI-Powered Vulnerability Scanner")
    st.markdown("""
    Welcome to the **Ad_VulnxAI**! This tool combines cutting-edge AI and cybersecurity techniques to:
    - Detect vulnerabilities in your network.
    - Predict risks and exploitation likelihood.
    - Generate detailed reports for remediation.
    """)
    
    st.markdown("""
    ### Key Features:
    - **Data Analysis**: Upload and analyze datasets for anomalies and risks.
    - **Network Scanning**: Perform Nmap scans to identify open ports and services.
    - **Intrusion Detection**: Monitor network traffic for suspicious activity.
    - **Automated Reporting**: Generate detailed vulnerability reports.
    """)

    # Login button to transition to the main app
    if st.button(" Click Me 🙂"):
        st.session_state.logged_in = True
        st.rerun()

def validate_dataset(df):
    """
    Validate the uploaded dataset for required columns and data types.
    """
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("The dataset must contain at least one numeric column for analysis.")
    
    return True


def main_app():
    """
    Enhanced main app logic with improved model integration and visualization
    """
    st.title("AI-Powered Vulnerability Scanner 🐞")
    
    # Initialize session states
    if "reports" not in st.session_state:
        st.session_state.reports = []
    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = None
    if "training_history" not in st.session_state:
        st.session_state.training_history = None
    if "scan_data" not in st.session_state:  # New storage for all scan data
        st.session_state.scan_data = {}

    def calculate_overall_risk(scan_data):
        """Calculate comprehensive risk score across all scans"""
        risk_score = 0
        max_score = 0
        
        for scan_type, results in scan_data.items():
            if 'vulnerability_scores' in results:
                risk_score += results['vulnerability_scores'].mean()
                max_score += 1
            if 'risk_scores' in results:
                risk_score += results['risk_scores'].mean()
                max_score += 1
                
        final_score = (risk_score / max_score) if max_score > 0 else 0
        
        return f"""
        - **Overall Risk Score**: {final_score:.2f}/1.00
        - **Risk Level**: {'Critical' if final_score > 0.8 else 'High' if final_score > 0.6 else 'Medium' if final_score > 0.4 else 'Low'}
        - **Affected Systems**: {len(scan_data)} different scan targets
        """

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Analysis", "🌐 Network Scanner", "🚨 Intrusion Detection", "📄 Reports"])

    with tab1:
        st.header("📂 Advanced Data Analysis")
        st.write("""
        **Upload your dataset** (CSV, Excel, JSON, or log files) for:
        - Advanced data preprocessing and feature engineering
        - Deep learning-based anomaly detection
        - Risk prediction with uncertainty estimation
        - Interactive visualizations and insights
        """)

        # Dataset Selection with improved error handling
        st.subheader("📁 Select a Dataset")
        dataset_folder = "dataset"
        
        try:
            available_datasets = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
            selected_dataset = st.selectbox("Choose a dataset:", available_datasets)
        except FileNotFoundError:
            st.warning(f"Dataset folder '{dataset_folder}' not found. Creating it...")
            os.makedirs(dataset_folder, exist_ok=True)
            available_datasets = []
            selected_dataset = None

        uploaded_file = st.file_uploader(
            "Upload a dataset (CSV, Excel, JSON, Log):", 
            type=["csv", "xlsx", "json", "log", "txt"]
        )

        @st.cache_data
        def load_dataset(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return None

        # Initialize dataset variable
        df = None

        # Enhanced dataset loading with progress bar
        if uploaded_file:
            try:
                with st.spinner("Loading and validating dataset..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith(('.log', '.txt')):
                        log_content = uploaded_file.getvalue().decode("utf-8")
                        df = parse_log_file(log_content)
                    
                    if df is not None:
                        st.success("Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        elif selected_dataset:
            dataset_path = os.path.join(dataset_folder, selected_dataset)
            df = load_dataset(dataset_path)

        # Display dataset info and statistics
        if df is not None:
            st.subheader("📊 Dataset Overview")
            st.write("Dataset Shape:", df.shape)
            st.write("Missing Values:", df.isnull().sum().sum())

            # Add data profiling option
            if st.checkbox("Show Detailed Data Profile"):
                st.write("Dataset Statistics:")
                st.write(df.describe())
                
                # Display correlation matrix using Streamlit's built-in Plotly support
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    st.write("Correlation Matrix:")
                    corr_matrix = df.select_dtypes(include=[np.number]).corr()

                    # Create a heatmap using Plotly
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='Viridis'
                    ))

                    fig.update_layout(
                        title='Correlation Matrix',
                        xaxis_title='Features',
                        yaxis_title='Features'
                    )

                    st.plotly_chart(fig)


        # Enhanced workflow execution
        if st.button("Run Advanced Analysis Workflow"):
            try:
                if df is None:
                    raise ValueError("Please upload or select a dataset first.")

                # Validate dataset
                validate_dataset(df)

                def plot_training_history(history):
                    """
                    Plot training history metrics
                    """
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history['train_loss'], name='Training Loss'))
                    fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
                    fig.add_trace(go.Scatter(y=history['val_f1'], name='Validation F1'))
                    fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Value')
                    return fig

                def plot_risk_scores_with_uncertainty(risks, uncertainties):
                    """
                    Plot risk scores with uncertainty bands
                    """
                    fig = go.Figure()

                    # Add risk scores
                    fig.add_trace(go.Scatter(
                        y=risks,
                        mode='lines',
                        name='Risk Score',
                        line=dict(color='blue')
                    ))

                    # Add uncertainty bands
                    fig.add_trace(go.Scatter(
                        y=risks + uncertainties,
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        y=risks - uncertainties,
                        mode='lines',
                        name='Lower Bound',
                        line=dict(width=0),
                        fillcolor='rgba(0, 0, 255, 0.2)',
                        fill='tonexty',
                        showlegend=False
                    ))

                    fig.update_layout(
                        title='Risk Scores with Uncertainty',
                        yaxis_title='Risk Score',
                        xaxis_title='Sample Index'
                    )
                    return fig

                def plot_risk_distribution(risk_scores):
                    """
                    Plot risk score distribution
                    """
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=risk_scores,
                        nbinsx=30,
                        name='Risk Distribution'
                    ))
                    fig.update_layout(
                        title='Risk Score Distribution',
                        xaxis_title='Risk Score',
                        yaxis_title='Count'
                    )
                    return fig

                # Data Cleansing with progress tracking
                st.subheader("🧹 Advanced Data Cleansing")
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Cleaning data...")
                progress_bar.progress(20)
                df_cleaned = clean_data(df, drop_na=True, impute_missing=True, handle_outliers=True)
                st.text("Cleaned Data:")
                st.dataframe(df_cleaned.head())
                
                status_text.text("Preprocessing data...")
                progress_bar.progress(40)
                numeric_data = preprocess_data(df_cleaned, encode_categorical=True)
                st.text("Preprocessed Data:")
                

                # Model Training with enhanced visualization
                st.subheader("🤖 Advanced Risk Prediction")
                status_text.text("Training AI model...")
                progress_bar.progress(60)

                # Generate synthetic labels for demonstration
                labels = np.random.randint(0, 2, size=(numeric_data.shape[0],))

                # Train the enhanced model
                model, scaler, training_history, X_val, y_val = train_model(numeric_data, labels)

                # Store training history
                st.session_state.training_history = training_history

                # Add this after the training history visualization in the Streamlit app
                # ===================== Model Evaluation Section =====================
                st.subheader("📈 Model Performance Evaluation")

                # Create tabs for different types of evaluation
                eval_tab1, eval_tab2 = st.tabs(["Training History", "Comprehensive Metrics"])

                with eval_tab1:
                    if st.session_state.training_history:
                        st.write("### Model Training Progress")
                        
                        # Create a Plotly figure for training history
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=st.session_state.training_history['train_loss'],
                            name='Training Loss',
                            mode='lines+markers',
                            line=dict(color='#FF4B4B')
                        ))
                        fig.add_trace(go.Scatter(
                            y=st.session_state.training_history['val_loss'],
                            name='Validation Loss',
                            mode='lines+markers',
                            line=dict(color='#0068C9')
                        ))
                        fig.add_trace(go.Scatter(
                            y=st.session_state.training_history['val_f1'],
                            name='Validation F1 Score',
                            mode='lines+markers',
                            line=dict(color='#00C897'),
                            yaxis='y2'
                        ))
                        
                        fig.update_layout(
                            title='Training Metrics History',
                            xaxis_title='Epochs',
                            yaxis_title='Loss Value',
                            yaxis2=dict(
                                title='F1 Score',
                                overlaying='y',
                                side='right',
                                rangemode='tozero'
                            ),
                            hovermode="x unified",
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No training history available for visualization")

                with eval_tab2:
                        # Perform comprehensive evaluation
                        metrics = evaluate_model_comprehensive(model, X_val, y_val, scaler)  # Remove .numpy()
                        
                        # Create a metrics dataframe for better visualization
                        metrics_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC'],
                            'Value': [
                                metrics['accuracy'],
                                metrics['precision'],
                                metrics['recall'],
                                metrics['f1_score'],
                                metrics['roc_auc'],
                                metrics['pr_auc']
                            ]
                        })

                        # Display metrics in columns with color coding
                        st.write("### Comprehensive Performance Metrics")
                        
                        cols = st.columns(3)
                        metric_styles = {
                            'Accuracy': ('#00C897', '🟢'),
                            'Precision': ('#0068C9', '🔵'),
                            'Recall': ('#FF4B4B', '🔴'),
                            'F1 Score': ('#FFC700', '🟡'),
                            'ROC AUC': ('#7D3C98', '🟣'),
                            'PR AUC': ('#FF6B6B', '❤️')
                        }

                        for idx, row in metrics_df.iterrows():
                            with cols[idx % 3]:
                                color, icon = metric_styles[row['Metric']]
                                st.markdown(f"""
                                <div style="
                                    padding: 1rem;
                                    border-radius: 0.5rem;
                                    background: {color}10;
                                    border-left: 4px solid {color};
                                    margin-bottom: 1rem;
                                ">
                                    <div style="font-size: 0.8rem; color: {color}; margin-bottom: 0.5rem;">
                                        {icon} {row['Metric']}
                                    </div>
                                    <div style="font-size: 1.2rem; font-weight: bold; color: {color};">
                                        {row['Value']:.3f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Add confusion matrix visualization
                        st.write("### Confusion Matrix")
                        from sklearn.metrics import confusion_matrix
                        import seaborn as sns
                        import matplotlib.pyplot as plt

                        y_pred = model(torch.tensor(scaler.transform(X_val), dtype=torch.float32))
                        y_pred = (y_pred > 0.5).float().numpy()
                        cm = confusion_matrix(y_val, y_pred)

                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=['Low Risk', 'High Risk'],
                                    yticklabels=['Low Risk', 'High Risk'])
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        st.pyplot(fig)

                # Plot training history if available
                if training_history:
                    st.write("Model Training History:")
                    fig = plot_training_history(training_history)
                    if fig:
                        st.plotly_chart(fig)
                    else:
                        st.warning("No training history to plot.")

                # Make predictions with uncertainty
                status_text.text("Making predictions...")
                progress_bar.progress(80)
                mean_predictions, uncertainty = predict_with_uncertainty(model, numeric_data, scaler)


                # Ensure predictions are valid before plotting
                if mean_predictions is not None and uncertainty is not None:
                    # Add predictions and uncertainty to dataframe
                    df_cleaned['Risk Score'] = mean_predictions
                    df_cleaned['Uncertainty'] = uncertainty

                    # Visualize results
                    st.subheader("📊 Advanced Risk Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_risk_scores_with_uncertainty(
                            df_cleaned['Risk Score'], 
                            df_cleaned['Uncertainty']
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No Risk Score plot available.")
                    with col2:
                        fig = plot_risk_distribution(df_cleaned['Risk Score'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No Risk Distribution plot available.")
                else:
                    st.warning("No predictions to display.")


                # Anomaly Detection
                st.subheader("🔍 Advanced Anomaly Detection")
                status_text.text("Detecting anomalies...")
                progress_bar.progress(90)
                
                detector = AnomalyDetector(algorithm="isolation_forest", contamination=0.05)
                anomalies = detector.detect_anomalies(numeric_data)
                df_cleaned['Anomaly'] = ['Yes' if a == -1 else 'No' for a in anomalies]

                # Visualize anomalies
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_anomaly_distribution(df_cleaned),
                        use_container_width=True
                    )
                with col2:
                    st.plotly_chart(
                        plot_scatter_anomalies(df_cleaned, 'Risk Score'),
                        use_container_width=True
                    )

                status_text.text("Analysis completed successfully!")
                st.success("Advanced analysis workflow completed!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                logging.error(f"Analysis error: {e}")
                
            finally:
                # Clear progress indicators
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()


    with tab2:
        st.header("🌐 Network Scanner")
        st.write("""
        **Network Scanning** uses Nmap and Scapy to:
        - Perform advanced scans on a target IP address.
        - Identify open ports, services, and operating systems.
        - Detect vulnerabilities and assess exploitation likelihood using AI.
        """)

        # Input for target IP and scan type
        target_ip = st.text_input("Enter Target IP for Network Scan", "8.8.8.8")
        scan_type = st.selectbox(
            "Select Scan Type",
            ["Quick Scan", "Service Detection", "OS Detection", "Aggressive Scan", "Full Scan", "ARP Scan", "TCP SYN Scan", "ICMP Ping Sweep", "Vulnerability Scan", "Service Version Detection"],
            help="Choose the type of scan to perform on the target IP."
        )

         # Custom ports input for TCP SYN Scan
        if scan_type == "TCP SYN Scan":
            important_ports = {
                21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
                80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS",
                3306: "MySQL", 3389: "RDP", 8080: "HTTP-Alt"
            }
            selected_ports = st.multiselect(
                "Select Ports to Scan",
                options=list(important_ports.keys()),
                format_func=lambda x: f"{x} ({important_ports[x]})",
                default=[22, 80, 443]
            )

        if st.button("Start Network Scan"):
            st.session_state.scan_data[scan_type] = {}
            st.info(f"Starting {scan_type} on {target_ip}...")
            scanner = NmapScanner()
            with st.spinner("Scanning the network..."):
                try:
                    if scan_type in ["Quick Scan", "Service Detection", "OS Detection", "Aggressive Scan", "Full Scan"]:
                        # Perform Nmap scan
                        nmap_result = scanner.scan_network(target_ip, scan_type)
                        time.sleep(2)  # Simulate scanning time
                        st.success("Nmap scan completed!")

                        st.subheader("Nmap Scan Results")
                        st.json(nmap_result)

                        # Display detailed scan information
                        if target_ip in nmap_result['scan']:
                            st.subheader("Detailed Scan Information")
                            scan_info = nmap_result['scan'][target_ip]
                            st.write(f"Hostname: {scan_info.get('hostnames', [{}])[0].get('name', 'N/A')}")
                            st.write(f"State: {scan_info['status']['state']}")

                            # Display open ports and services
                            if 'tcp' in scan_info:
                                st.write("Open TCP Ports:")
                                ports = []
                                states = []
                                services = []
                                for port, port_info in scan_info['tcp'].items():
                                    ports.append(port)
                                    states.append(port_info['state'])
                                    services.append(port_info['name'])
                                
                                # Create DataFrame with correct column names
                                port_df = pd.DataFrame({"Port": ports, "State": states, "Service": services})
                                st.session_state.port_df = port_df  # Store port_df in session state
                                
                                st.dataframe(port_df)

                                # Visualize open ports
                                st.subheader("📊 Open Ports Visualization")
                                if "Port" in port_df.columns:
                                    # Bar chart for open ports
                                    st.plotly_chart(plot_bar_chart(port_df, "Port", "Service", "Open Ports Distribution"))
                                    # Pie chart for service distribution
                                    service_counts = port_df["Service"].value_counts()
                                    st.plotly_chart(plot_pie_chart(service_counts, "Service Distribution"))
                                else:
                                    st.error("Column 'Port' not found in the DataFrame.")

                                # Predict vulnerability likelihood
                                st.subheader("🔮 Vulnerability Prediction")
                                #vulnerability_scores = predict_vulnerability(model, port_df , scaler)
                                vulnerability_scores = predict_vulnerability_mock(port_df)
                                port_df["Vulnerability Score"] = vulnerability_scores

                                st.dataframe(port_df)

                                # Visualize vulnerability scores
                                st.subheader("📊 Vulnerability Score Visualization")
                                if "Vulnerability Score" in port_df.columns:
                                    st.plotly_chart(plot_histogram(port_df, "Vulnerability Score", "Vulnerability Score Distribution"))
                                else:
                                    st.error("Column 'Vulnerability Score' not found in the DataFrame.")

                                # Predict exploitation likelihood
                                st.subheader("🔮 Exploitation Likelihood")
                                exploitation_scores = predict_exploitation_likelihood_mock(port_df)
                                port_df["Exploitation Likelihood"] = exploitation_scores

                                st.dataframe(port_df)

                                # Visualize exploitation likelihood
                                st.subheader("📊 Exploitation Likelihood Visualization")
                                if "Exploitation Likelihood" in port_df.columns:
                                    st.plotly_chart(plot_histogram(port_df, "Exploitation Likelihood", "Exploitation Likelihood Distribution"))
                                else:
                                    st.error("Column 'Exploitation Likelihood' not found in the DataFrame.")
                            else:
                                st.write("No open TCP ports found.")

                            # Display OS detection results
                            if 'osmatch' in scan_info:
                                st.write("OS Detection Results:")
                                os_matches = []
                                for os_match in scan_info['osmatch']:
                                    os_matches.append({"name": os_match['name'], "accuracy": os_match['accuracy']})
                                os_df = pd.DataFrame(os_matches)
                                st.dataframe(os_df)

                                # Visualize OS detection
                                st.subheader("📊 OS Detection Visualization")
                                if "accuracy" in os_df.columns:
                                    st.plotly_chart(plot_bar_chart(os_df, "name", "accuracy", "OS Detection Accuracy"))
                                else:
                                    st.error("Column 'accuracy' not found in the DataFrame.")
                            else:
                                st.write("No OS information detected.")

                            # Save results
                            st.session_state.scan_data[scan_type] = {
                                'nmap_result': nmap_result,
                                'port_df': port_df,
                                'os_df': os_df,
                                'vulnerability_scores': vulnerability_scores,
                                'exploitation_scores': exploitation_scores
                            }

                        else:
                            st.error("No scan results found for the target IP.")

                    elif scan_type == "ARP Scan":
                        # Perform ARP scan
                        live_hosts = scanner.arp_scan(target_ip)
                        st.success("ARP scan completed!")

                        st.subheader("ARP Scan Results")
                        live_hosts_df = pd.DataFrame(live_hosts)
                        st.dataframe(live_hosts_df)

                        # Visualize live hosts
                        st.subheader("📊 Live Hosts Visualization")
                        if "IP" in live_hosts_df.columns:
                            st.plotly_chart(plot_bar_chart(live_hosts_df, "IP", "MAC", "Live Hosts Distribution"))
                        else:
                            st.error("Column 'IP' not found in the DataFrame.")
                        # Save results
                        st.session_state.scan_data[scan_type] = {
                            'live_hosts_df': live_hosts_df,
                            'risk_scores': predict_exploitation_likelihood_mock(live_hosts_df)
                    }

                    elif scan_type == "TCP SYN Scan":
                        if not selected_ports:
                            st.error("Please select at least one port to scan.")
                        else:
                            # Perform TCP SYN scan
                            open_ports = scanner.tcp_syn_scan(target_ip, selected_ports)
                            st.success("TCP SYN scan completed!")

                            st.subheader("TCP SYN Scan Results")
                            if open_ports:
                                open_ports_df = pd.DataFrame({
                                    "Port": open_ports,
                                    "Service": [important_ports.get(port, "Unknown") for port in open_ports]
                                })
                                st.dataframe(open_ports_df)

                                # Visualize open ports
                                st.subheader("📊 Open Ports Visualization")
                                if "Port" in open_ports_df.columns:
                                    # Bar chart for open ports
                                    st.plotly_chart(plot_bar_chart(open_ports_df, "Port", "Service", "Open Ports Distribution"))
                                else:
                                    st.error("Column 'Port' not found in the DataFrame.")
                            else:
                                st.write("No open ports found.")

                    elif scan_type == "ICMP Ping Sweep":
                        # Perform ICMP ping sweep
                        is_up = scanner.icmp_ping_sweep(target_ip)
                        st.success("ICMP ping sweep completed!")

                        st.subheader("ICMP Ping Sweep Results")
                        st.write(f"Host {target_ip} is {'up' if is_up else 'down'}.")

                    elif scan_type == "Vulnerability Scan":
                        # Perform vulnerability scan
                        vulnerabilities = scanner.vulnerability_scan(target_ip)
                        st.success("Vulnerability scan completed!")

                        st.subheader("Vulnerability Scan Results")
                        if vulnerabilities:
                            vuln_df = pd.DataFrame(vulnerabilities)
                            st.dataframe(vuln_df)

                            # Visualize vulnerabilities
                            st.subheader("📊 Vulnerability Visualization")
                            if "Port" in vuln_df.columns:
                                # Bar chart for vulnerabilities by port
                                st.plotly_chart(plot_bar_chart(vuln_df, "Port", "Vulnerability", "Vulnerability Distribution by Port"))
                                
                                # Display detailed vulnerability descriptions
                                st.subheader("📄 Vulnerability Details")
                                for index, row in vuln_df.iterrows():
                                    with st.expander(f"Port {row['Port']} - {row['Vulnerability']}"):
                                        st.write(f"**Description:** {row['Description']}")
                            else:
                                st.error("Column 'Port' not found in the DataFrame.")
                            # Save results
                            st.session_state.scan_data[scan_type] = {
                                'vuln_df': vuln_df,
                                'risk_scores': predict_exploitation_likelihood_mock(vuln_df)
                    }
                        else:
                            st.write("No vulnerabilities detected.")

                    elif scan_type == "Service Version Detection":
                        # Perform service version detection
                        services = scanner.service_version_detection(target_ip)
                        st.success("Service version detection completed!")

                        st.subheader("Service Version Detection Results")
                        if services:
                            service_df = pd.DataFrame(services)
                            st.dataframe(service_df)

                            # Visualize services
                            st.subheader("📊 Service Version Visualization")
                            if "Port" in service_df.columns:
                                # Bar chart for services by port
                                st.plotly_chart(plot_bar_chart(service_df, "Port", "Service", "Service Distribution by Port"))
                                
                                # Display detailed service information
                                st.subheader("📄 Service Details")
                                for index, row in service_df.iterrows():
                                    with st.expander(f"Port {row['Port']} - {row['Service']}"):
                                        st.write(f"**Product:** {row['Product']}")
                                        st.write(f"**Version:** {row['Version']}")
                                        st.write(f"**Extra Info:** {row['Extra Info']}")
                                        st.write(f"**CPE:** {row['CPE']}")
                            else:
                                st.error("Column 'Port' not found in the DataFrame.")

                            # Save results
                            st.session_state.scan_data[scan_type] = {
                                'service_df': service_df,
                                'risk_scores': predict_exploitation_likelihood_mock(service_df)
                            }
                        else:
                            st.write("No services detected.")

                except Exception as e:
                    st.error(f"An error occurred during the network scan: {e}")
                    logging.error(f"Network scan error: {e}")

    with tab3:
        st.title("🚨 IDS System")
        # Initialize IDS
        if "ids" not in st.session_state:
            st.session_state.ids = IntrusionDetectionSystem()

        # Initialize monitoring state
        if "monitoring_active" not in st.session_state:
            st.session_state.monitoring_active = False

        def toggle_monitoring():
            """
            Toggle the monitoring state between start and stop.
            """
            st.session_state.monitoring_active = not st.session_state.monitoring_active

        def monitor_traffic():
            """
            Simulate real-time monitoring of network traffic and display alerts.
            """
            alert_placeholder = st.empty()  # Placeholder to display alerts

            while st.session_state.monitoring_active:
                # Simulate traffic and detect intrusions
                st.session_state.ids.simulate_traffic()
                alerts = st.session_state.ids.detect_intrusions(st.session_state.ids.traffic_data)

                # Display alerts in real-time
                if not alerts.empty:
                    alert_placeholder.dataframe(alerts)
                else:
                    alert_placeholder.write("No intrusions detected.")

                # Simulate a delay for real-time monitoring
                time.sleep(5)  # Adjust the delay as needed

        # Display description
        st.write("""
        **Intrusion Detection System (IDS)** monitors network traffic for:
        - Suspicious activity (e.g., port scans, brute force attacks).
        - Real-time alerts for potential intrusions.

        **How to Use:**
        - Click the button below to **Start Monitoring**.
        - Click the same button again to **Stop Monitoring**.
        """)

        # Toggle button for monitoring
        if st.session_state.monitoring_active:
            if st.button("Stop Monitoring"):
                toggle_monitoring()
                st.write("Monitoring stopped.")
        else:
            if st.button("Start Monitoring"):
                toggle_monitoring()
                st.write("Monitoring started...")
                monitor_traffic()

    with tab4:
        st.header("📄 Reports")
        st.write("""
        **View and manage generated vulnerability reports.**
        Reports can be generated from scan data and downloaded as text files.
        """)

        # Generate a new report
        if st.button("Generate New Report"):
            if 'scan_data' in st.session_state and st.session_state.scan_data:
                with st.spinner("Generating report..."):
                    report = generate_vulnerability_report(st.session_state.scan_data)
                    # Initialize reports list if not present
                    if 'reports' not in st.session_state:
                        st.session_state.reports = []
                    st.session_state.reports.append(report)
                    st.success("Report generated and saved!")
            else:
                st.error("No scan data available. Please perform a network scan first.")

        # Display and allow downloading of saved reports
        if 'reports' in st.session_state and st.session_state.reports:
            st.subheader("Saved Reports")
            for i, report in enumerate(st.session_state.reports):
                with st.expander(f"Report {i + 1}"):
                    st.markdown(report)  # Display in Markdown format
                    st.download_button(
                        label="Download Report as TXT",
                        data=report,
                        file_name=f"vulnerability_report_{i + 1}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No reports available. Generate a report to view it here.")

def main():
    st.set_page_config(page_title="🛡️ Ad_VulnxAI", layout="wide")

    # Load custom CSS
    def load_css(file_name):
        """Load and apply custom CSS from a file."""
        if Path(file_name).exists():
            with open(file_name) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    load_css("style.css")

    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Show intro page or main app based on login state
    if not st.session_state.logged_in:
        show_intro()
    else:
        main_app()

    # Add footer
    st.markdown(
        """
        <div class="footer">
            <p>Developed with ❤️ by <a href="https://aadi-web8.vercel.app/" target="_blank">Aditya Pandey</a></p>
            <p>© 2025 Advanced AI-Powered Vulnerability Scanner</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()