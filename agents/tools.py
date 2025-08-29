import logging
import os
import threading

import chromadb
import PyPDF2

from google.adk.models.lite_llm import LiteLlm

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

class ChromaClientSingleton:
    """A thread-safe singleton class for managing ChromaDB client instances.

    This class ensures only one ChromaDB client instance exists throughout the application
    lifecycle using the Singleton pattern with thread-safety mechanisms.

    Attributes:
        _instance: The single ChromaDB client instance.
        _lock: A threading lock for thread-safe instance creation.
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton ChromaDB client instance.

        Returns:
            The singleton ChromaDB client instance with initialized collection.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = chromadb.PersistentClient(path="./chromadb")
        return cls._instance

class EmbeddingManagerSingleton:
    """EmbeddingManagerSingleton class.

    This class is a singleton that manages the embedding model. It uses the SentenceTransformer
    library to encode text into embeddings. The model is loaded from the "thenlper/gte-small"
    checkpoint. The singleton pattern ensures that only one instance of the model is created
    and used throughout the application.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SentenceTransformer("thenlper/gte-small")
        return cls._instance


def chunk_pdf(pdf_path: str, chunk_size: int = 5000, chunk_overlap: int = 500) -> list[dict[str]]:
    """Split PDF into semantic chunks with basic metadata.
    
    Args:
        pdf_path: The path to the PDF file to be chunked.
        chunk_size: The size of each chunk (default: 1000 tokens).
        chunk_overlap: The number of overlapping tokens between chunks (default: 200 tokens).

    Returns:
        list[dict[str]]: A list of dictionaries, where each dictionary contains:
            - text: The content of the chunk
            - source: The source identifier for citation purposes
            - chunk_id: The chunk unique identifier
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"Page {page_num+1}: {page_text}\n"

    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk_text = text[i:i + chunk_size]
        if len(chunk_text) > 100:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": os.path.basename(pdf_path),
                    "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{len(chunks)}"
                }
            )

    return chunks


def retrieve_context(query: str, top_k: int = 3, similarity_threshold: float = 0.7) -> list[dict]:
    """Enhanced context retrieval with semantic filtering.

    Args:
        query: The search query text to find relevant documents.
        top_k: The number of most relevant documents to retrieve (default: 3).
        similarity_threshold: The minimum cosine similarity score for a document to be included (0-1, default: 0.7).

    Returns:
        list[dict]: Filtered list of relevant documents
    """
    logger.info(f"Retrieving context. Query: {query}")
    chroma_client = ChromaClientSingleton.get_instance()
    collection = chroma_client.get_or_create_collection(name="visio_docs")
    
    embedder = EmbeddingManagerSingleton.get_instance()
    query_embedding = embedder.encode(query, normalize_embedding=True).tolist()
    logger.info(f"Query embedding: {query_embedding}")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2, # Get more results for filtering
        include=["documents", "metadatas", "distances"]
    )
    logger.info(f"Found {len(results)} results before filtering. ({results['distances'][:3]})")
    
    logger.info(f"Results: {results}")
    
    filtered_results = []
    for doc, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - distance
        if similarity >= similarity_threshold:
            filtered_results.append({
                "page_content": doc,
                "source": metadata.get("source", "unknown"),
                "similarity": round(similarity, 3)
            })
    logger.info(f"Found {filtered_results} results after filtering. ({filtered_results[:3]})")
    logger.info(f"Example: {filtered_results[0]}")

    return filtered_results[:top_k]

def expand_query(query: str) -> str:
    """Expand a query using a LLM."""
    expander = LiteLlm(model="openai/gpt-4o-mini")
    expanded_query_response = expander.generate_content_async(
        f"Expand the following query: {query}"
    ).text
    logger.info(f"Expanded query: {expanded_query_response}")
    return query