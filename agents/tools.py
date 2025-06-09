import chromadb

client = chromadb.Client()
collection = client.create_collection(name="visio_docs")
collection.upsert(
    documents=[
        "Visio has 100 GPUs.",
        "Visio has 1000 CPUs.",
        "Visio has 10000 storage units.",
    ],
    ids=["gpu", "cpu", "storage"],
)

def retrieve_context(query: str, top_k: int = 1) -> list[dict]:
    """Search ChromaDB and return the top‑k docs most relevant to *query*.
    Each dict must include a `page_content` and `source` key so the agent can cite it."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"]
