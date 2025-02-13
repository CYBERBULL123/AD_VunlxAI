import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import time
from pathlib import Path
import logging
from data_processing.data_cleaning import clean_data, preprocess_data
from data_processing.log_parser import parse_log_file
from models.anomaly_detection import AnomalyDetector
from models.risk_prediction import train_model, predict_vulnerability, predict_exploitation_likelihood , predict_vulnerability_mock , predict_exploitation_likelihood_mock
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
    Main app logic after login.
    """
    st.title("AI-Powered Vulnerability Scanner 🐞")
    # Initialize session state for reports
    if "reports" not in st.session_state:
        st.session_state.reports = []

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Analysis", "🌐 Network Scanner", "🚨 Intrusion Detection", "📄 Reports"])

    with tab1:
        st.header("📂 Data Analysis")
        st.write("""
        **Upload your dataset** (CSV, Excel, JSON, or log files) to:
        - Clean and preprocess the data.
        - Detect anomalies using Isolation Forest.
        - Predict risk scores using a PyTorch-based neural network.
        - Visualize results with interactive charts.
        """)

        st.warning("Ensure your dataset contains numeric features and labels.")

        # Dataset Selection
        st.subheader("📁 Select a Dataset")
        dataset_folder = "dataset"  # Folder containing your datasets
        available_datasets = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
        selected_dataset = st.selectbox("Choose a dataset:", available_datasets)
        uploaded_file = st.file_uploader("Upload a dataset (CSV, Excel, JSON, Log):", type=["csv", "xlsx", "json", "log", "txt"])

        @st.cache_data
        def load_dataset(file_path):
            return pd.read_csv(file_path)

        # Initialize dataset variable
        df = None

        # Handle dataset loading
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.log') or uploaded_file.name.endswith('.txt'):
                    log_content = uploaded_file.getvalue().decode("utf-8")
                    df = parse_log_file(log_content)
            except Exception as e:
                st.error(f"Error loading uploaded dataset: {e}")
        elif selected_dataset:
            dataset_path = os.path.join(dataset_folder, selected_dataset)
            df = load_dataset(dataset_path)

        # Display raw data if available
        if df is not None:
            st.subheader("📄 Raw Data")
            st.dataframe(df)

        # Add a button to run the entire workflow
        if st.button("Run Full Workflow"):
            try:
                # Validate dataset
                validate_dataset(df)

                # Display raw data
                st.subheader("📄 Raw Data")
                st.write("Here's a preview of your uploaded data:")
                st.dataframe(df.head())

                # Data Cleansing
                st.subheader("🧹 Data Cleansing")
                with st.spinner("Cleaning data..."):
                    df = clean_data(df, drop_na=True, impute_missing=True, handle_outliers=True)
                st.success("Data cleansing completed!")
                st.write("Cleansed Data:")
                st.dataframe(df.head())

                # Preprocess data
                st.subheader("🔧 Data Preprocessing")
                with st.spinner("Preprocessing data..."):
                    numeric_data = preprocess_data(df, encode_categorical=True)
                st.success("Data preprocessing completed!")
                st.write("Preprocessed Numeric Data (First 5 rows):")
                st.write(numeric_data[:5])

                # Train model and predict risk
                st.subheader("🤖 Risk Prediction")
                with st.spinner("Training the AI model..."):
                    labels = np.random.randint(0, 2, size=(numeric_data.shape[0],))  # Mock labels
                    model, scaler = train_model(numeric_data, labels)
                st.success("Model training completed!")

                # Predict risk levels
                risks = predict_vulnerability(model, numeric_data, scaler)
                df["Risk Score"] = risks.flatten()

                # Display risk scores
                st.write("Risk Scores:")
                st.dataframe(df.head())

                # Visualize risk scores
                st.subheader("📊 Risk Score Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_risk_scores(df), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_histogram(df, "Risk Score", "Distribution of Risk Scores"), use_container_width=True)

                # Anomaly detection
                st.subheader("🔍 Anomaly Detection")
                with st.spinner("Detecting anomalies..."):
                    true_labels = np.random.choice([1, -1], size=numeric_data.shape[0], p=[0.95, 0.05])
                    detector = detector = AnomalyDetector(algorithm="isolation_forest", contamination=0.05)
                    anomalies = detector.detect_anomalies(numeric_data)
                    df["Anomaly"] = ["Yes" if a == -1 else "No" for a in anomalies]
                st.success("Anomaly detection completed!")

                # Display anomalies
                st.write("Anomalies Detected:")
                st.dataframe(df.head())

                 # Evaluate performance
                metrics = detector.evaluate_anomalies(numeric_data, true_labels)
                st.write("Anomaly Detection Metrics:")
                st.write(metrics)

                # Visualize anomalies
                st.subheader("📊 Anomaly Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_anomaly_distribution(df), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_scatter_anomalies(df, "Risk Score"), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_heatmap(df, "Risk Score", "Anomaly"), use_container_width=True)
                    with col2:
                        fig = detector.plot_anomalies(numeric_data, feature_1=0, feature_2=1)
                        st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred during data analysis: {e}")
                logging.error(f"Data analysis error: {e}")

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
            # Define important ports with their names
            important_ports = {
                21: "FTP",
                22: "SSH",
                23: "Telnet",
                25: "SMTP",
                53: "DNS",
                80: "HTTP",
                110: "POP3",
                143: "IMAP",
                443: "HTTPS",
                3306: "MySQL",
                3389: "RDP",
                8080: "HTTP-Alt"
            }

            # Multi-select box for ports
            selected_ports = st.multiselect(
                "Select Ports to Scan",
                options=list(important_ports.keys()),
                format_func=lambda x: f"{x} ({important_ports[x]})",
                default=[22, 80, 443]
            )

        if st.button("Start Network Scan"):
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
        """)

        if st.button("Generate New Report"):
            if "port_df" in st.session_state and st.session_state.port_df is not None:
                with st.spinner("Generating report..."):
                    report = generate_vulnerability_report(st.session_state.port_df)
                    st.session_state.reports.append(report)  # Save report in session state
                    st.success("Report generated and saved!")
            else:
                st.error("No scan data available. Please perform a network scan first.")

        # Display saved reports
        if st.session_state.reports:
            st.subheader("Saved Reports")
            for i, report in enumerate(st.session_state.reports):
                with st.expander(f"Report {i + 1}"):
                    st.write(report)
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