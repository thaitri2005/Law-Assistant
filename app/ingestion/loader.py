import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.google import GoogleDriveReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

def ingest_documents():
    """
    Ingests documents from Google Drive into Pinecone.
    """
    # 1. Configure Embeddings (E5-Large)
    # Pinecone does not generate embeddings automatically in this setup; we generate them locally.
    print("Loading embedding model (multilingual-e5-large)...")
    # Use device="cpu" to avoid GPU memory issues if CUDA is present but limited
    # Use trust_remote_code=True if needed, though E5 usually doesn't need it
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large",
        device="cpu" 
    )

    # 2. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "law-assistant-index")
    
    # Create index if not exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024, # E5-large dimension is 1024
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Load Data from Google Drive
    # Note: You need credentials.json for this to work
    loader = GoogleDriveReader(
        credentials_path="credentials.json",
        token_path="token.json"
    )
    
    # Load specific folder
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    print(f"Loading documents from folder: {folder_id}")
    documents = loader.load_data(folder_id=folder_id)
    
    print(f"Found {len(documents)} documents.")
    if not documents:
        print("No documents found! Check your Folder ID or if the folder is empty.")
        return

    # 3. Index Data
    print("Indexing documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()
