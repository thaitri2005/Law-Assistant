import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def check_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in .env")
        return

    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        print("Existing Pinecone Indexes:")
        for idx in indexes:
            print(f"- Name: {idx.name}, Status: {idx.status['state']}, Host: {idx.host}")
            
        index_name = os.getenv("PINECONE_INDEX_NAME", "law-assistant-index")
        if any(idx.name == index_name for idx in indexes):
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"\nStats for '{index_name}':")
            print(stats)
        else:
            print(f"\nIndex '{index_name}' does not exist.")

    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")

if __name__ == "__main__":
    check_pinecone()
