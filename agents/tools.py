import chromadb
import threading


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
                    cls._instance = chromadb.Client()
                    collection = cls._instance.create_collection(name="visio_docs")
                    collection.upsert(
                        documents=[
                            "Visio has 100 GPUs.",
                            "Visio has 1000 CPUs.",
                            "Visio has 10000 storage units.",
                        ],
                        ids=["gpu", "cpu", "storage"],
                    )
        return cls._instance


def retrieve_context(query: str, top_k: int = 1) -> list[dict]:
    """Search ChromaDB and retrieve the most relevant documents matching the query.

    This tool searches through a ChromaDB collection and returns the top-k most semantically
    relevant documents. The results are formatted as a list of dictionaries, each containing
    the document content and its source identifier for proper citation.

    Args:
        query: The search query text to find relevant documents.
        top_k: The number of most relevant documents to return (default: 1).

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - page_content: The content of the matched document
            - source: The source identifier for citation purposes
    """
    global chroma_client

    chroma_client = ChromaClientSingleton.get_instance()
    collection = chroma_client.get_collection(name="visio_docs")
    results = collection.query(query_texts=[query], n_results=top_k)
    return [{"page_content": doc, "source": src} for doc, src in zip(results["documents"][0], results["ids"][0])]
