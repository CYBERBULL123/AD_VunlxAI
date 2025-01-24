# reporting/threat_intelligence.py

from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import logging
from functools import lru_cache
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a tool for threat intelligence lookup
class ThreatIntelligenceTool(BaseTool):
    name: str = "Threat Intelligence Lookup"  # Add type annotation
    description: str = "Look up the latest threat intelligence for a given vulnerability."  # Add type annotation

    def _run(self, vulnerability_name: str) -> str:
        """
        Simulate a threat intelligence lookup (replace with actual API calls).
        """
        try:
            # Simulate a threat intelligence lookup
            logger.info(f"Looking up threat intelligence for: {vulnerability_name}")
            return f"Threat intelligence for {vulnerability_name}: High risk, actively exploited in the wild."
        except Exception as e:
            logger.error(f"Error looking up threat intelligence: {e}")
            return f"Unable to retrieve threat intelligence for {vulnerability_name}."

# Initialize the agent
tools = [ThreatIntelligenceTool()]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Cache results to avoid redundant API calls
@lru_cache(maxsize=100)
def get_threat_intelligence(vulnerability_name: str) -> str:
    """
    Get dynamic insights for a given vulnerability using LangChain.
    """
    try:
        logger.info(f"Fetching threat intelligence for: {vulnerability_name}")
        result = agent.run(f"Look up threat intelligence for {vulnerability_name}.")
        logger.info(f"Threat intelligence result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error fetching threat intelligence: {e}")
        return f"Unable to fetch threat intelligence for {vulnerability_name}."