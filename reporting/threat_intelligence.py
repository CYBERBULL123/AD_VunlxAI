import requests
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from functools import lru_cache
import streamlit as st

# Configure API Keys from Streamlit secrets
gemini_key = st.secrets["api_keys"]["gemini"]
nvd_api_key = st.secrets["api_keys"]["nvd"]  # Optional for NVD
virustotal_api_key = st.secrets["api_keys"]["virustotal"]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=gemini_key,
    temperature=0.7,
    max_tokens=3000,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Threat Intelligence Tool
class ThreatIntelligenceTool(BaseTool):
    name: str = "Threat Intelligence Lookup"
    description: str = "Look up advanced threat intelligence for a vulnerability or CVE ID using NVD and VirusTotal."

    def _run(self, vulnerability_input: str) -> str:
        try:
            # Determine if input is a CVE ID or a vulnerability name
            if vulnerability_input.upper().startswith("CVE-"):
                cve_id = vulnerability_input.upper()
            else:
                search_url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={vulnerability_input}"
                response = requests.get(search_url)
                if response.status_code == 200:
                    data = response.json()
                    if data["totalResults"] > 0:
                        cve_id = data["vulnerabilities"][0]["cve"]["id"]
                    else:
                        return f"Final Answer: No CVE found for '{vulnerability_input}'."
                else:
                    return f"Final Answer: Error searching NVD: {response.status_code}."

            # Fetch NVD details
            detail_url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
            response = requests.get(detail_url)
            if response.status_code == 200:
                data = response.json()
                cve_data = data["vulnerabilities"][0]["cve"]
                description = cve_data["descriptions"][0]["value"]
                cvss_score = (
                    cve_data["metrics"]["cvssMetricV31"][0]["cvssData"]["baseScore"]
                    if "cvssMetricV31" in cve_data["metrics"]
                    else "Not Available"
                )
            else:
                return f"Final Answer: Error fetching CVE details from NVD: {response.status_code}."

            # Fetch VirusTotal insights
            vt_search_url = f"https://www.virustotal.com/api/v3/search?query={cve_id}"
            headers = {"X-Api-Key": virustotal_api_key}
            response = requests.get(vt_search_url, headers=headers)
            vt_info = "No related threats found on VirusTotal."
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    malicious_count = sum(
                        1 for item in data["data"] if item["attributes"]["last_analysis_stats"]["malicious"] > 0
                    )
                    vt_info = (
                        f"Found {len(data['data'])} items (files/URLs) related to {cve_id}, "
                        f"{malicious_count} of which are flagged as malicious."
                    )
            else:
                vt_info = f"Error searching VirusTotal: {response.status_code}."

            report = (
                f"Vulnerability: {cve_id}\n"
                f"Description: {description}\n"
                f"CVSS Score: {cvss_score}\n"
                f"VirusTotal Insights: {vt_info}"
            )
            return f"Final Answer: {report}"
        except Exception as e:
            logger.error(f"Error in ThreatIntelligenceTool: {e}")
            return f"Final Answer: Unable to retrieve threat intelligence for '{vulnerability_input}' due to an error."

# Initialize the Agent
tools = [ThreatIntelligenceTool()]
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=1
)

# Cached Threat Intelligence Function
@lru_cache(maxsize=100)
def get_threat_intelligence(vulnerability_input: str) -> str:
    try:
        logger.info(f"Fetching threat intelligence for: {vulnerability_input}")
        result = agent.run(f"Look up threat intelligence for {vulnerability_input}.")
        logger.info(f"Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error fetching threat intelligence: {e}")
        return f"Unable to fetch threat intelligence for '{vulnerability_input}'."