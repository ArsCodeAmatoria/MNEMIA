"""
Simple memory manager for MNEMIA
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

def setup_memory():
    """Setup basic memory connection"""
    try:
        client = QdrantClient("localhost", port=6333)
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Memory setup successful")
        return client, encoder
    except Exception as e:
        print(f"❌ Memory setup failed: {e}")
        return None, None

if __name__ == "__main__":
    setup_memory()

