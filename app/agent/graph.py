from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os

# Initialize Pinecone & Embeddings globally to avoid reloading
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "law-assistant-index")
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Configure LlamaIndex to use the same embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)

# Create the retriever
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = index.as_retriever(similarity_top_k=3)

# Define the state of the agent
class AgentState(TypedDict):
    messages: list
    context: str

# --- Nodes ---

def retrieve_node(state: AgentState):
    """
    Node to retrieve documents from Pinecone.
    """
    query = state["messages"][-1].content
    print(f"Retrieving info for: {query}")
    
    try:
        results = retriever.retrieve(query)
        context = "\n\n".join([node.get_content() for node in results])
        if not context:
            context = "No relevant documents found in the legal database."
    except Exception as e:
        context = f"Error retrieving documents: {str(e)}"
    
    return {"context": context}

def web_search_node(state: AgentState):
    """
    Node to perform web search if internal docs are insufficient.
    """
    query = state["messages"][-1].content
    print(f"Searching web for: {query}")
    
    try:
        search = GoogleSearchAPIWrapper()
        web_results = search.run(query)
    except Exception as e:
        web_results = f"Error performing web search: {str(e)}"
        
    return {"context": web_results}

def generate_node(state: AgentState):
    """
    Node to generate the final answer using LLM.
    """
    messages = state["messages"]
    context = state.get("context", "")
    
    prompt = f"""
    You are a Vietnamese Law Assistant. Answer the user's question based on the context below.
    If the context is empty, say you don't know.
    
    Context: {context}
    
    Question: {messages[-1].content}
    """
    
    # Using Gemini 2.5 Pro (Correcting model name - NOT 1.5)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    response = llm.invoke(prompt)
    
    return {"messages": [response]}

# --- Conditional Logic ---

def route_query(state: AgentState) -> Literal["retrieve", "web_search"]:
    """
    Decides whether to use internal RAG or Web Search.
    """
    last_message = state["messages"][-1].content.lower()
    
    if "news" in last_message or "update" in last_message:
        return "web_search"
    else:
        return "retrieve"

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

# Add edges
workflow.set_conditional_entry_point(
    route_query,
    {
        "retrieve": "retrieve",
        "web_search": "web_search"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
