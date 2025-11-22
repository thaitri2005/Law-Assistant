import sys
import os

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from app.agent.graph import app as agent_app
from langchain_core.messages import HumanMessage

# Load env vars
load_dotenv()

def test_agent():
    print("--- Starting Test Run ---")
    
    # 1. Test Query (General/Web Search)
    query = "What are the latest updates on Vietnam Land Law in 2024?"
    print(f"\nUser: {query}")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    try:
        result = agent_app.invoke(inputs)
        response = result["messages"][-1].content
        print(f"\nAgent: {response}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Check if GOOGLE_API_KEY and GOOGLE_CSE_ID are set in .env")

if __name__ == "__main__":
    test_agent()
