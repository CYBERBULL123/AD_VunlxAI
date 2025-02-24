import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from reporting.threat_intelligence import get_threat_intelligence  # Assuming this is in a separate file

# Configure API Key for Gemini LLM
gemini_key = st.secrets["api_keys"]["gemini"]

# Initialize GEMINI LLM for analysis and remediation
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=gemini_key,
    temperature=0.7,
    max_tokens=3000,  # Allow for detailed responses
)

def analyze_vulnerability_data(data_str):
    """Analyze scan data and generate a summary."""
    analysis_prompt = PromptTemplate(
        input_variables=["data"],
        template="""
        Analyze the following scan data and provide a detailed summary:
        {data}

        Include:
        - Total number of findings (e.g., open ports, vulnerabilities).
        - Most critical issues (e.g., open ports on critical services, vulnerabilities with high severity).
        - Likelihood of exploitation based on the data.
        - Brief recommendations for immediate action.
        """
    )
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    return analysis_chain.run(data=data_str)

def generate_remediation_recommendations(analysis_result):
    """Generate remediation recommendations based on analysis."""
    remediation_prompt = PromptTemplate(
        input_variables=["analysis_result"],
        template="""
        Based on this analysis, provide detailed remediation recommendations:
        {analysis_result}

        Include:
        - Immediate actions for critical issues (e.g., close ports, apply patches).
        - Long-term strategies (e.g., firewall rules, regular scans).
        - References to best practices (e.g., NIST SP 800-53, OWASP Top 10).
        """
    )
    remediation_chain = LLMChain(llm=llm, prompt=remediation_prompt)
    return remediation_chain.run(analysis_result=analysis_result)

def generate_vulnerability_report(scan_data):
    """
    Generate a comprehensive vulnerability report for all scan types.

    Args:
        scan_data (dict): Dictionary containing scan results by scan type (e.g., 'Quick Scan': {'port_df': ...}).

    Returns:
        str: Markdown-formatted report string.
    """
    if not scan_data:
        return "No scan data available to generate a report."

    report = "# 🛡️ Vulnerability Report\n\n"
    all_data_str = ""  # Aggregated data for analysis
    critical_issues = []  # Track critical findings for threat intelligence

    # Process each scan type
    for scan_type, data in scan_data.items():
        report += f"## {scan_type} Results:\n"

        if scan_type in ["Quick Scan", "Service Detection", "OS Detection", "Aggressive Scan", "Full Scan"]:
            if data.get('port_df') is not None:
                port_df = data['port_df']
                all_data_str += f"{scan_type} - Open Ports:\n{port_df.to_string()}\n\n"
                report += "### Open Ports:\n" + port_df.to_markdown(index=False) + "\n\n"
                # Identify critical ports/services
                if "Port" in port_df.columns:
                    for _, row in port_df.iterrows():
                        port = row.get("Port")
                        service = row.get("Service", "Unknown")
                        if port in [21, 22, 23, 25, 80, 110, 143, 443, 3306, 3389]:  # Common risky ports
                            critical_issues.append({"Port": port, "Service": service})
            if data.get('os_df') is not None:
                os_df = data['os_df']
                all_data_str += f"{scan_type} - OS Detection:\n{os_df.to_string()}\n\n"
                report += "### OS Detection:\n" + os_df.to_markdown(index=False) + "\n\n"

        elif scan_type == "ARP Scan":
            if data.get('live_hosts_df') is not None:
                live_hosts_df = data['live_hosts_df']
                all_data_str += f"{scan_type} - Live Hosts:\n{live_hosts_df.to_string()}\n\n"
                report += "### Live Hosts:\n" + live_hosts_df.to_markdown(index=False) + "\n\n"

        elif scan_type == "TCP SYN Scan":
            if data.get('open_ports_df') is not None:
                open_ports_df = data['open_ports_df']
                all_data_str += f"{scan_type} - Open Ports:\n{open_ports_df.to_string()}\n\n"
                report += "### Open Ports:\n" + open_ports_df.to_markdown(index=False) + "\n\n"
                if "Port" in open_ports_df.columns:
                    for _, row in open_ports_df.iterrows():
                        port = row.get("Port")
                        service = row.get("Service", "Unknown")
                        if port in [21, 22, 23, 25, 80, 110, 143, 443, 3306, 3389]:
                            critical_issues.append({"Port": port, "Service": service})

        elif scan_type == "ICMP Ping Sweep":
            if 'is_up' in data:
                status = "up" if data['is_up'] else "down"
                all_data_str += f"{scan_type} - Host Status: {status}\n\n"
                report += f"### Host Status:\nHost is {status}.\n\n"

        elif scan_type == "Vulnerability Scan":
            if data.get('vuln_df') is not None:
                vuln_df = data['vuln_df']
                all_data_str += f"{scan_type} - Vulnerabilities:\n{vuln_df.to_string()}\n\n"
                report += "### Detected Vulnerabilities:\n" + vuln_df.to_markdown(index=False) + "\n\n"
                if "Vulnerability" in vuln_df.columns:
                    critical_issues.extend(vuln_df.to_dict('records'))

        elif scan_type == "Service Version Detection":
            if data.get('service_df') is not None:
                service_df = data['service_df']
                all_data_str += f"{scan_type} - Service Versions:\n{service_df.to_string()}\n\n"
                report += "### Service Versions:\n" + service_df.to_markdown(index=False) + "\n\n"
                if "Service" in service_df.columns:
                    for _, row in service_df.iterrows():
                        service = row.get("Service", "Unknown")
                        critical_issues.append({"Service": service})

    # Generate threat intelligence for critical issues
    threat_intel = []
    if not critical_issues:
        threat_intel.append("No critical issues detected for threat intelligence lookup.")
    else:
        for issue in critical_issues:
            identifier = issue.get("Service") or issue.get("Vulnerability") or f"Port {issue.get('Port', 'Unknown')}"
            intel = get_threat_intelligence(identifier)
            threat_intel.append(f"- {identifier}: {intel}")

    # Generate analysis and remediation
    analysis_result = analyze_vulnerability_data(all_data_str if all_data_str else "No detailed scan data provided.")
    remediation_result = generate_remediation_recommendations(analysis_result)

    # Compile the final report
    report += (
        "## 🔍 Analysis Summary:\n"
        f"{analysis_result}\n\n"
        "## 🚨 Threat Intelligence:\n"
        + "\n".join(threat_intel) + "\n\n"
        "## 🛠️ Remediation Recommendations:\n"
        f"{remediation_result}\n\n"
        "## 📊 Detailed Scan Data:\n"
        f"{all_data_str}"
    )

    return report