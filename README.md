# Vietnamese Law Assistant AI (Agentic RAG)

## Overview
This project is an AI-powered Law Assistant designed to answer questions about Vietnamese law. It combines **Retrieval-Augmented Generation (RAG)** with **Agentic capabilities**.

## Architecture

The system is built as an **Agent-Augmented RAG** pipeline. It doesn't just retrieve documents; it can decide whether to use internal legal documents, search the web for recent updates, or combine both.

### High-Level Flow
1.  **User Query**: User asks a question via the API/UI.
2.  **Agent Router (LangGraph)**: The agent analyzes the query.
    *   *Is it about specific legal codes?* -> Route to **Vector Store (Pinecone)**.
    *   *Is it about recent news or general info?* -> Route to **Web Search**.
    *   *Is it complex?* -> Break down into sub-queries.
3.  **Retrieval**:
    *   **Internal**: Fetch relevant chunks from Pinecone (indexed from Google Drive).
    *   **External**: Perform a web search (e.g., Tavily/DuckDuckGo).
4.  **Synthesis**: The LLM generates an answer with citations.

## Tech Stack & Tool Choices

### 1. Orchestration & Logic
*   **LangGraph**: For building the stateful agent workflow (cyclic graphs). This allows the agent to "loop" (e.g., if retrieved docs aren't good enough, rewrite query and try again).
*   **LangChain**: The underlying framework for chains and prompts.

### 2. Data Ingestion (ETL)
*   **LlamaIndex**: Used for its superior data connectors and parsing capabilities.
    *   `GoogleDriveReader`: To pull PDFs/Docs from the specified Google Drive folder.
    *   `SentenceSplitter`: To chunk text appropriately for legal contexts.

### 3. Vector Database
*   **Pinecone**: Serverless vector database. Scalable and fast.
    *   *Why?* Managed service, good performance, easy to scale for production.

### 4. Web Search (Agent Tool)
*   **Google Custom Search API**:
    *   *Why?* Reliable and comprehensive search results directly from Google.

### 5. LLM & Embeddings
*   **LLM**: Google Gemini 1.5 Pro.
*   **Embeddings**: `intfloat/multilingual-e5-large` (HuggingFace). Run locally via `sentence-transformers`.

### 6. Backend & Deployment
*   **FastAPI**: To expose the agent as a REST API.
*   **Docker**: For containerization (Cloud-ready).

## Folder Structure
```
Law-Assistant/
├── app/
│   ├── agent/          # LangGraph agent logic (nodes, edges)
│   ├── ingestion/      # LlamaIndex pipelines (Drive -> Pinecone)
│   ├── api/            # FastAPI routes
│   └── main.py         # Entry point
├── docs/               # Documentation
├── requirements.txt    # Dependencies
└── .env                # Environment variables (API Keys)
```

## Setup
1.  Install dependencies: `pip install -r requirements.txt`
2.  Set up `.env` with `GOOGLE_API_KEY`, `PINECONE_API_KEY`, `GOOGLE_DRIVE_CREDENTIALS`.
