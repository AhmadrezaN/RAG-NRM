"""
Optimized Local Vector Retriever Module for Qdrant
"""

from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embedder import get_embeddings_batch

COLLECTION_NAME = "islamic_religious_movements"
STORAGE_PATH = "./qdrant_data"

_CLIENT_INSTANCE: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Returns a singleton instance of the local persistent Qdrant database."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = QdrantClient(path=STORAGE_PATH)
    return _CLIENT_INSTANCE


def close_qdrant_client():
    """Explicitly closes the Qdrant client connection before exit."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is not None:
        _CLIENT_INSTANCE.close()
        _CLIENT_INSTANCE = None


def embed_query(query: str) -> list[float]:
    """Embeds a single user query using the required Nomic prefix."""
    formatted_query = f"search_query: {query.strip()}"
    vectors = get_embeddings_batch([formatted_query])
    return vectors[0]


def retrieve_relevant_chunks(
    query: str, 
    top_k: int = 4, 
    chapter_filter: Optional[str] = None,
    score_threshold: float = 0.35
) -> list[dict]:
    """
    Searches Qdrant for top_k relevant chunks with optional filtering 
    and similarity score thresholding.
    """
    client = get_qdrant_client()

    # 1. Generate query vector
    query_vector = embed_query(query)

    # 2. Build metadata filter if specified
    query_filter = None
    if chapter_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.chapter",
                    match=MatchValue(value=chapter_filter)
                )
            ]
        )

    # 3. Perform similarity search
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        score_threshold=score_threshold
    )

    # 4. Format payload output
    retrieved_chunks = []
    for hit in search_response.points:
        payload = hit.payload
        metadata = payload.get("metadata", payload)

        retrieved_chunks.append({
            "score": round(hit.score, 4),
            "content": payload.get("content", ""),
            "chapter": metadata.get("chapter", "Unknown"),
            "subsection": metadata.get("subsection", "Unknown"),
            "start_page": metadata.get("start_page", 0),
            "end_page": metadata.get("end_page", 0),
            "chunk_id": payload.get("chunk_id", "N/A"),
        })

    return retrieved_chunks


if __name__ == "__main__":
    test_query = "Who is Ibn Taymiyya"
    
    print(f"\nSearching for: '{test_query}'\n" + "=" * 60)
    matches = retrieve_relevant_chunks(test_query, top_k=4)

    if not matches:
        print("No matching chunks found above the relevance threshold.")
    else:
        for i, match in enumerate(matches, start=1):
            print(f"\n[Match #{i}] Similarity Score: {match['score']}")
            print(f"Location: {match['chapter']} -> {match['subsection']} (Pages {match['start_page']}-{match['end_page']})")
            print(f"Content Preview:\n{match['content'][:]}...")
            print("-" * 60)

    # Explicitly close connection to prevent garbage collection error on shutdown
    close_qdrant_client()