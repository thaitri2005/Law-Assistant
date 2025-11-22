from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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
    
    # TODO: Connect to Pinecone retriever here
    # results = retriever.invoke(query)
    # context = "\n".join([doc.page_content for doc in results])
    
    context = "Mock context: Article 1 of Vietnamese Civil Code..." 
    return {"context": context}

def web_search_node(state: AgentState):
    """
    Node to perform web search if internal docs are insufficient.
    """
    query = state["messages"][-1].content
    print(f"Searching web for: {query}")
    
    # TODO: Connect to Tavily or DuckDuckGo here
    web_results = "Mock web result: Recent amendment to..."
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
    
    # Using Gemini 2.5 Pro (2.5 is available now, 1.5 is retired)
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
