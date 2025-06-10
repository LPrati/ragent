import os
import threading

import chromadb
import PyPDF2

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


def retrieve_context(query: str, top_k: int = 15) -> list[dict]:
    """Search ChromaDB and retrieve the most relevant documents matching the query.

    This tool searches through a ChromaDB collection and returns the top-k most semantically
    relevant documents. The results are formatted as a list of dictionaries, each containing
    the document content and its source identifier for proper citation.

    Args:
        query: The search query text to find relevant documents.
        top_k: The number of most relevant documents to return (default: 15).

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - page_content: The content of the matched document
            - source: The source identifier for citation purposes
    """
    global chroma_client

    chroma_client = ChromaClientSingleton.get_instance()
    collection = chroma_client.get_or_create_collection(name="visio_docs")
    results = collection.query(query_texts=[query], n_results=top_k)
    return [{"page_content": doc, "source": src} for doc, src in zip(results["documents"][0], results["ids"][0])]


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
            chunks.append({
                "text": chunk_text,
                "source": os.path.basename(pdf_path),
                "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{len(chunks)}"
            })

    return chunks
