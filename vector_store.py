"""
Local Vector Store Module using Embedded Qdrant (No Docker Required)
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from embedder import load_and_embed_pdf

# Configuration
COLLECTION_NAME = "islamic_religious_movements"
VECTOR_SIZE = 768  # Matches nomic-embed-text dimensions
STORAGE_PATH = "./qdrant_data"


def get_qdrant_client() -> QdrantClient:
    """Initializes a local, persistent file-based Qdrant client in ./qdrant_data."""
    return QdrantClient(path=STORAGE_PATH)


def initialize_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME, recreate: bool = False):
    """
    Creates or verifies the vector collection and configures payload indexes for metadata filtering.
    """
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if recreate and exists:
        print(f"Recreating existing collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE, 
                distance=Distance.COSINE
            ),
        )
        # Build payload indexes for metadata filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.section_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.chapter",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("✓ Collection and payload indexes created successfully.")
    else:
        print(f"✓ Collection '{collection_name}' verified.")


def store_embeddings_in_qdrant(embedded_chunks: list[dict], batch_size: int = 64):
    """
    Stores vectors and complete metadata payloads into the local Qdrant database using batch upserts.
    """
    client = get_qdrant_client()
    initialize_collection(client, collection_name=COLLECTION_NAME, recreate=True)

    total_chunks = len(embedded_chunks)
    print(f"\nPreparing {total_chunks} points for indexing in '{COLLECTION_NAME}'...")

    points = []
    for item in embedded_chunks:
        chunk_id_str = item["chunk_id"]
        
        # Generate a deterministic UUID based on chunk_id string
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id_str))

        # Comprehensive payload including raw text, ID, and structural metadata
        payload = {
            "chunk_id": chunk_id_str,
            "content": item["content"],
            "metadata": item["metadata"]
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=item["embedding"],
                payload=payload,
            )
        )

    print(f"Writing vectors to disk in '{STORAGE_PATH}' (Batch size: {batch_size})...")
    
    # Batch upsert into Qdrant
    for i in range(0, total_chunks, batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"Progress: [{min(i + batch_size, total_chunks)}/{total_chunks}] points written.")

    print(f"\n✅ SUCCESS: All {total_chunks} vector embeddings successfully indexed in {STORAGE_PATH}!")


def verify_qdrant_store():
    """Prints diagnostic statistics about the stored collection."""
    client = get_qdrant_client()
    info = client.get_collection(COLLECTION_NAME)
    
    print("\n" + "=" * 60)
    print("QDRANT VECTOR STORE DIAGNOSTICS")
    print("=" * 60)
    print(f"Collection Name: {COLLECTION_NAME}")
    print(f"Total Points:    {info.points_count}")
    print(f"Status:          {info.status}")
    print("=" * 60)


if __name__ == "__main__":
    print("Step 1: Extracting text chunks and generating Ollama embeddings...")
    embedded_chunks = load_and_embed_pdf()

    print("\nStep 2: Indexing into local Qdrant vector database...")
    store_embeddings_in_qdrant(embedded_chunks)

    print("\nStep 3: Running diagnostics...")
    verify_qdrant_store()