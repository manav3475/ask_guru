import sys
import glob
from pathlib import Path

# Add parent directory to path FIRST before importing config
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from logger import logger
from qdrant_client import QdrantClient, models
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def check_and_delete_collection(client: QdrantClient,
                                collection_name: str) -> bool:
    '''Check if collection exists or not and Delete if it already exists
    Args:
        client: Qdrant client object to setup a connection
        collection_name: Name of the collection

    Return:
        Boolean value. True if collections exists(and deleted) else False
    '''

    collections = [i.name for i in client.get_collections().collections]
    if collection_name in collections:
        client.delete_collection(collection_name=collection_name)
        return True
    else:
        return False


def embedding_and_indexing(client: QdrantClient,
                           data_dir: str,
                           collection_name: str,
                           chunk_size: int,
                           chunk_overlap: int,
                           vector_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                           vector_size: int = 384,
                           sparse_embedding_model: str = "distilbert-base-uncased") -> None:
    '''Create a new collection. Read the PDF documents and create embeddings
    and store it in Qdrant vectorDB
    Args:
        client: Qdrant client object to setup a connection
        data_dir: Data directory where PDF documents are stored
        collection_name: Name of the collection to be created
        chunk_size: Chunk size for splitting the documents
        chunk_overlap: Overlapping while splitting the documents
        vector_embedding_model: Model to be used for dense vector creation (default: sentence-transformers/all-MiniLM-L6-v2)
        vector_size: Length of the vector_embedding_model (default: 384)
        sparse_embedding_model: Model to be used for sparse embedding creation (default: distilbert-base-uncased)
    '''
    # Check if collection exists, delete if it exists
    
    status = check_and_delete_collection(client=client, collection_name=collection_name)
    
    if status:
        logger.debug(
            f"Collection {collection_name} already exists. Deleted the collection.")

    # Looping through all PDF files in the data directory
    all_docs = []
    for filepath in glob.glob(data_dir):
        logger.debug(f"Reading the PDF file {filepath}")
        loader = PyPDFLoader(filepath)
        docs = loader.lazy_load()
        all_docs.extend(docs)
    
 
    logger.info("Chunking process has started")
   

    # Chunking and splitting the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    documents = text_splitter.split_documents(all_docs)



if __name__ == "__main__":
    qdrant_client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=120.0
    )

    embedding_and_indexing(
        client=qdrant_client,
        data_dir=config.DATA_DIR,
        collection_name=config.COLLECTION_NAME,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
