import sys
import os

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.graph import app as agent_app
from app.ingestion.loader import ingest_documents
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

app = FastAPI(title="Vietnamese Law Assistant AI")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Law Assistant AI is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/sync")
async def sync_drive():
    """
    Triggers the ingestion process to pull latest files from Google Drive.
    """
    try:
        ingest_documents()
        return {"status": "success", "message": "Ingestion complete. Index updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Endpoint to chat with the Law Assistant Agent.
    """
    try:
        inputs = {"messages": [HumanMessage(content=request.query)]}
        result = agent_app.invoke(inputs)
        
        # Extract the last message from the agent
        last_message = result["messages"][-1].content
        return {"response": last_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
