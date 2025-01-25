# reporting/report_generator.py

from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Use Gemini
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
from reporting.threat_intelligence import get_threat_intelligence

# Load environment variables from .env file
load_dotenv()

# Function to load the Google API key
def load_google_api_key():
    """
    Load the Google API key from Streamlit secrets or .env file.
    """
    # Check Streamlit secrets first
    if "api_keys" in st.secrets and "gemini" in st.secrets["api_keys"]:
        print("GOOGLE_API_KEY found in Streamlit secrets.")
        return st.secrets["api_keys"]["gemini"]
    
    # Fall back to .env file for local development
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("GOOGLE_API_KEY found in .env file.")
        return google_api_key
    
    # If no key is found, raise an error
    raise ValueError("GOOGLE_API_KEY not found in Streamlit secrets or .env file. Please add it.")

# Load the Google API key
GOOGLE_API_KEY = load_google_api_key()

# Get the Google API key from the environment
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# # Initialize Gemini
# genai.configure(api_key=GOOGLE_API_KEY)  # Pass the API key to configure Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# Define a LangChain agent for report generation
def generate_vulnerability_report(data):
    """
    Generate a detailed vulnerability report using LangChain and Gemini.
    """
    # Step 1: Analyze the data
    analysis_prompt = PromptTemplate(
        input_variables=["data"],
        template="""
        Analyze the following vulnerability data and provide a summary:
        {data}

        Key points to include:
        - Total number of vulnerabilities.
        - Most critical vulnerabilities (based on risk score).
        - Exploitation likelihood for critical vulnerabilities.
        - Recommendations for remediation.
        """
    )
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    analysis_result = analysis_chain.run(data=data.to_string())

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

    # Step 3: Get dynamic insights for critical vulnerabilities
    critical_vulnerabilities = data[data["Vulnerability Score"] > 0.8]  # Example threshold
    threat_intelligence = []
    for _, row in critical_vulnerabilities.iterrows():
        vulnerability_name = row["Service"]
        intelligence = get_threat_intelligence(vulnerability_name)
        threat_intelligence.append(f"- {vulnerability_name}: {intelligence}")

    # Step 4: Generate the final report
    report = (
        "🛡️ Vulnerability Report 🛡️\n\n"
        "🔍 Analysis Summary:\n"
        f"{analysis_result}\n\n"
        "🚨 Threat Intelligence:\n"
        + "\n".join(threat_intelligence) + "\n\n"
        "🛠️ Remediation Recommendations:\n"
        f"{remediation_result}\n\n"
        "📊 Detailed Data:\n"
        f"{data.to_string()}"
    )

    return report