"""
Local Vector Embedding Module using Ollama REST API (nomic-embed-text)
"""

import requests
import numpy as np
import pymupdf as fitz
from cleaner import slice_and_chunk_subsections
from config import BOOK_STRUCTURE, PDF_PATH

# Configuration
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings_batch(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Sends a batch of text strings to Ollama's /api/embed endpoint 
    and returns a list of vector embeddings.
    """
    # Nomic embedding convention: prefix texts with 'search_document: ' for indexing
    formatted_inputs = [f"search_document: {t}" for t in texts]

    payload = {
        "model": model,
        "input": formatted_inputs
    }

    try:
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]
    except requests.exceptions.RequestException as e:
        print(f"\n[Error] Failed to connect to Ollama service at {OLLAMA_EMBED_URL}.")
        print("Ensure Ollama is running and 'nomic-embed-text' is pulled (`ollama pull nomic-embed-text`).")
        raise e


def generate_embeddings_for_chunks(chunks: list[dict], batch_size: int = 16) -> list[dict]:
    """
    Processes structured chunks in batches, attaches vector embeddings,
    and assigns a unique chunk_id to each item.
    """
    total_chunks = len(chunks)
    print(f"\nGenerating vectors for {total_chunks} chunks using '{EMBEDDING_MODEL}' (Batch size: {batch_size})...")

    embedded_chunks = []

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c["content"] for c in batch_chunks]

        # Get batch vector embeddings from Ollama
        vectors = get_embeddings_batch(batch_texts)

        # Attach each vector and assign a deterministic chunk_id
        for idx, (chunk, vector) in enumerate(zip(batch_chunks, vectors)):
            global_index = i + idx
            meta = chunk["metadata"]
            
            chunk_id = f"sec{meta.get('section_id', 0)}_chk{global_index}"

            chunk_entry = {
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "metadata": meta,
                "embedding": vector
            }
            embedded_chunks.append(chunk_entry)

        processed_count = min(i + batch_size, total_chunks)
        print(f"Progress: [{processed_count}/{total_chunks}] chunks embedded. (Vector size: {len(vectors[0])})")

    return embedded_chunks


def load_and_embed_pdf() -> list[dict]:
    """
    Helper function to execute PDF parsing via cleaner.py and generate embeddings.
    """
    print("Step 1: Parsing PDF into structured chunks...")
    doc = fitz.open(PDF_PATH)
    raw_chunks = slice_and_chunk_subsections(doc, BOOK_STRUCTURE, max_words=300, overlap_words=40)
    doc.close()

    print(f"--> Successfully extracted {len(raw_chunks)} clean chunks.")

    print("\nStep 2: Generating vector embeddings...")
    embedded_dataset = generate_embeddings_for_chunks(raw_chunks, batch_size=16)
    
    return embedded_dataset


def verify_embedded_dataset(embedded_dataset: list[dict], sample_query: str = "What is the definition of NRM?"):
    """
    Runs verification checks on the embedded dataset:
    1. Vector integrity & dimensionality check.
    2. ID uniqueness & section coverage.
    3. Quick cosine similarity test against a sample query.
    """
    print("\n" + "=" * 60)
    print("RUNNING EMBEDDING DATASET VERIFICATION")
    print("=" * 60)

    # 1. Vector Quality & Dimensionality Check
    invalid_vectors = [e for e in embedded_dataset if len(e["embedding"]) != 768]
    print(f"✓ Vector Integrity: {len(invalid_vectors)} invalid vectors found (Expected dimension: 768).")

    # 2. ID & Section Coverage Verification
    unique_ids = set(e["chunk_id"] for e in embedded_dataset)
    print(f"✓ Unique ID Check: {len(unique_ids)} / {len(embedded_dataset)} unique IDs generated.")

    sections_covered = sorted(list(set(e["metadata"]["section_id"] for e in embedded_dataset)))
    print(f"✓ Section Coverage: Covered Section IDs -> {sections_covered}")

    # 3. Quick Vector Similarity Test (Cosine Similarity)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\n--- Similarity Test for Query: '{sample_query}' ---")
    
    # Nomic convention: prefix query with 'search_query: '
    query_payload = {
        "model": EMBEDDING_MODEL, 
        "input": [f"search_query: {sample_query}"]
    }
    
    try:
        res = requests.post(OLLAMA_EMBED_URL, json=query_payload, timeout=30)
        res.raise_for_status()
        query_vector = res.json()["embeddings"][0]

        scores = []
        for entry in embedded_dataset:
            score = cosine_similarity(query_vector, entry["embedding"])
            scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)

        top_score, top_match = scores[0]
        print(f"Top Match Similarity Score: {top_score:.4f}")
        print(f"Matched Location:          {top_match['metadata']['chapter']} -> {top_match['metadata']['subsection']}")
        print(f"Matched Content Preview:   {top_match['content'][:]}")
        print("=" * 60)

    except Exception as e:
        print(f"[Warning] Could not execute similarity test query: {e}")


if __name__ == "__main__":
    # Execute full embedding and run verification checks
    embedded_dataset = load_and_embed_pdf()
    
    if embedded_dataset:
        verify_embedded_dataset(embedded_dataset)