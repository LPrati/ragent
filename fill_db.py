import os

from pathlib import Path

from agents.tools import chunk_pdf

if __name__ == "__main__":
    from agents.tools import ChromaClientSingleton

    chroma_client = ChromaClientSingleton.get_instance()
    collection = chroma_client.get_or_create_collection(name="visio_docs")

    for filepath in [
        "./documents/2025.05_6pager_Strategy_Company.pdf",
        "./documents/2025.05_6pager_Tactic_CoreGrowthSubway.pdf",
        "./documents/2025.05_6pager_Tactic_InternationalExpansion.pdf",
        "./documents/2025.05_6pager_Tactic_LateralExpansion.pdf",
    ]:
        chunks = chunk_pdf(filepath)
        source = os.path.basename(filepath)
        collection.upsert(
            documents=[
                chunk["text"] for chunk in chunks
            ],
            ids=[chunk["chunk_id"] for chunk in chunks],
            metadatas=[{"source": source} for chunk in chunks],
        )
