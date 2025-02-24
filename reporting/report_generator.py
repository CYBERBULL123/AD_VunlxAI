# reporting/report_generator.py

from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Use Gemini
import streamlit as st
from reporting.threat_intelligence import get_threat_intelligence
import pandas as pd

# Configure API Key
gemini_key = st.secrets["api_keys"]["gemini"]

# Initialize GEMINI LLM for reasoning tasks
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=gemini_key,
    temperature=0.7,
    max_tokens=3000,  # Allow for detailed responses
)

def generate_vulnerability_report(scan_data):
    """
    Generate a detailed vulnerability report using LangChain and Gemini based on scan data.

    Args:
        scan_data (dict): Dictionary of scan types and their data.

    Returns:
        str: A formatted vulnerability report.
    """
    if not scan_data:
        return "No scan data available to generate a report."

    report = "# 🛡️ Vulnerability Report\n\n"

    # Aggregate data for analysis
    all_data_str = ""
    critical_vulnerabilities = []

    # Process each scan type
    for scan_type, data in scan_data.items():
        report += f"## {scan_type} Results:\n"
        
        if scan_type in ["Quick Scan", "Service Detection", "OS Detection", "Aggressive Scan", "Full Scan"]:
            if data.get('port_df') is not None:
                port_df = data['port_df']
                all_data_str += f"{scan_type} - Open Ports:\n{port_df.to_string()}\n\n"
                report += "### Open Ports:\n" + port_df.to_markdown(index=False) + "\n\n"
                # Identify critical vulnerabilities
                if "Vulnerability Score" in port_df.columns:
                    critical = port_df[port_df["Vulnerability Score"] > 0.8]
                    critical_vulnerabilities.extend(critical.to_dict('records'))
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

        elif scan_type == "ICMP Ping Sweep":
            if 'is_up' in data:
                status = "up" if data['is_up'] else "down"
                all_data_str += f"{scan_type} - Host Status: {status}\n\n"
                report += f"Host is {status}.\n\n"

        elif scan_type == "Vulnerability Scan":
            if data.get('vuln_df') is not None:
                vuln_df = data['vuln_df']
                all_data_str += f"{scan_type} - Vulnerabilities:\n{vuln_df.to_string()}\n\n"
                report += "### Detected Vulnerabilities:\n" + vuln_df.to_markdown(index=False) + "\n\n"
                # Identify critical vulnerabilities
                if "Vulnerability" in vuln_df.columns:
                    critical_vulnerabilities.extend(vuln_df.to_dict('records'))

        elif scan_type == "Service Version Detection":
            if data.get('service_df') is not None:
                service_df = data['service_df']
                all_data_str += f"{scan_type} - Service Versions:\n{service_df.to_string()}\n\n"
                report += "### Service Versions:\n" + service_df.to_markdown(index=False) + "\n\n"

    # Step 1: Analyze the data using LangChain
    analysis_prompt = PromptTemplate(
        input_variables=["data"],
        template="""
        Analyze the following vulnerability data and provide a summary:
        {data}

        Key points to include:
        - Total number of vulnerabilities.
        - Most critical vulnerabilities (based on risk score or severity).
        - Exploitation likelihood for critical vulnerabilities.
        - Recommendations for remediation.
        """
    )
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    analysis_result = analysis_chain.run(data=all_data_str)

    # Step 2: Generate remediation recommendations
    remediation_prompt = PromptTemplate(
        input_variables=["analysis_result"],
        template="""
        Based on the following vulnerability analysis, provide detailed remediation recommendations:
        {analysis_result}

        Recommendations should include:
        - Immediate actions for critical vulnerabilities.
        - Long-term strategies to improve security posture.
        - References to industry best practices (e.g., NIST, OWASP).
        """
    )
    remediation_chain = LLMChain(llm=llm, prompt=remediation_prompt)
    remediation_result = remediation_chain.run(analysis_result=analysis_result)

    # Step 3: Get threat intelligence for critical vulnerabilities
    threat_intelligence = []
    for vuln in critical_vulnerabilities:
        vuln_name = vuln.get("Service") or vuln.get("Vulnerability", "Unknown")
        intelligence = get_threat_intelligence(vuln_name)
        threat_intelligence.append(f"- {vuln_name}: {intelligence}")

    # Step 4: Compile the final report
    report += (
        "## 🔍 Analysis Summary:\n"
        f"{analysis_result}\n\n"
        "## 🚨 Threat Intelligence:\n"
        + "\n".join(threat_intelligence) + "\n\n"
        "## 🛠️ Remediation Recommendations:\n"
        f"{remediation_result}\n\n"
        "## 📊 Detailed Scan Data:\n"
        f"{all_data_str}"
    )

    return report