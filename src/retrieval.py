from typing import List
import sys
from pathlib import Path

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.logger import logger

def retrieve_documents(client: QdrantClient,
                       query: str,
                       collection_name: str,
                       k: int=5,
                       vector_embedding_model: str="sentence-transformers/all-MiniLM-L6-v2",
                       sparse_embedding_model: str="Qdrant/bm25") -> List:
    """Retrieve Top k documents from the Qdrant vector DB

    Args:
        client: Qdrant client object to setup a connection
        query: Search query from users
        collection_name: Name of the collection
        k: Int to return Top K results (default: 5)
        vector_embedding_model: Model to be used for dense vector creation (default: sentence-transformers/all-MiniLM-L6-v2)
        sparse_embedding_model: Model to be used for sparse embedding creation (default: Qdrant/bm25)
    Returns:
        List of retrived documents. Size of the list is k
    """

    # Initialize embeddings (must match index.py)
    embeddings = HuggingFaceEmbeddings(model=vector_embedding_model)
    sparse_embeddings = FastEmbedSparse(model_name=sparse_embedding_model)

    logger.info("Connecting to VectorDB")

    # Connect to existing vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )

    results = vectorstore.similarity_search(
        query=query,
        k=k,  
    )

    logger.info(f"Retrieved the results for query: {query}")

    return results


if __name__ == "__main__":

    # Setup Qdrant client
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60.0
    )

    # Example usage
    query = "what is the ML ?"
    
    top_k = 5

    retrieved_docs = retrieve_documents(
        client=client,
        query=query,
        collection_name=config.COLLECTION_NAME,
        k=top_k
    )

    for i, doc in enumerate(retrieved_docs, 1):
        # print(f"Document {i}: {doc.page_content}\n")
        print(f"\nResult {i}:")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:300]}...")