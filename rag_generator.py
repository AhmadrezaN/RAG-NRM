"""
Local RAG Generation Module
Combines retriever.py context with local LLM generation via Ollama REST API.
Streams LLM thought process and final grounded response cleanly.
"""

import json
import requests
from retriever import retrieve_relevant_chunks, close_qdrant_client

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:4b"


def display_selected_chunks(chunks: list[dict]):
    """Prints full contents and metadata of top selected context chunks."""
    print("\n" + "=" * 65)
    print(f" RETRIEVED CONTEXT CHUNKS ({len(chunks)} Selected)")
    print("=" * 65)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n[CHUNK #{idx}] (Similarity Score: {chunk.get('score', 'N/A')})")
        print(f"Location : {chunk['chapter']} -> {chunk['subsection']} (Pages {chunk['start_page']}-{chunk['end_page']})")
        print("-" * 65)
        print(chunk["content"].strip())
        print("-" * 65)


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Formats retrieved vector chunks into a structured prompt."""
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        block = (
            f"--- Context Block {idx} ---\n"
            f"Location: {chunk['chapter']} -> {chunk['subsection']} (Pages {chunk['start_page']}-{chunk['end_page']})\n"
            f"Text:\n{chunk['content']}\n"
        )
        context_blocks.append(block)

    formatted_context = "\n".join(context_blocks)

    return f"""You are an academic research assistant answering questions based on the book "New Religious Movements in Islam".

Use ONLY the provided context blocks below to answer the user's question. If the context does not contain enough information to answer, state clearly that the text does not provide sufficient details.

CONTEXT:
{formatted_context}

QUESTION:
{query}

ANSWER (Include source page citations where applicable):"""


def generate_rag_answer(query: str, top_k: int = 3, stream: bool = True) -> str:
    """Retrieves chunks, displays context, and streams LLM thinking process + response."""
    print(f"\n[1/2] Retrieving top {top_k} relevant chunks from Qdrant...")
    chunks = retrieve_relevant_chunks(query, top_k=top_k)

    if not chunks:
        print("No relevant context found in the database.")
        return "No relevant context found in the database."

    display_selected_chunks(chunks)

    prompt = build_rag_prompt(query, chunks)
    
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": stream,
        "think": True
    }

    print(f"\n[2/2] Generating answer using '{LLM_MODEL}'...\n" + "=" * 65)

    full_response = ""
    thinking_phase = False
    response_started = False

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL, 
            json=payload, 
            stream=stream, 
            timeout=(10, None)
        )
        response.raise_for_status()

        if stream:
            for line in response.iter_lines():
                if not line:
                    continue
                    
                chunk_json = json.loads(line.decode("utf-8"))
                
                thinking_token = chunk_json.get("thinking", "")
                text_token = chunk_json.get("response", "")

                # 1. Handle Native Ollama thinking field
                if thinking_token:
                    if not thinking_phase:
                        print("[LLM THINKING PROCESS]:\n")
                        thinking_phase = True
                    print(thinking_token, end="", flush=True)

                # 2. Handle Text tokens (Answer phase or Inline <think> tags)
                if text_token:
                    # Detect inline <think> start
                    if "<think>" in text_token:
                        if not thinking_phase:
                            print("[LLM THINKING PROCESS]:\n")
                            thinking_phase = True
                        text_token = text_token.replace("<think>", "")

                    # Detect inline </think> end
                    if "</think>" in text_token:
                        text_token = text_token.replace("</think>", "")
                        thinking_phase = False

                    # Transition from thinking phase to final response phase
                    if thinking_phase and not thinking_token and text_token.strip():
                        # Standard text arrived while still in thinking_phase state -> end thinking
                        thinking_phase = False

                    if not thinking_phase:
                        if not response_started:
                            if thinking_token or "<think>" in text_token or thinking_phase:
                                print("\n\n" + "=" * 65 + "\n")
                            print("[LLM RESPONSE]:\n")
                            response_started = True

                        print(text_token, end="", flush=True)
                        full_response += text_token

            print("\n" + "=" * 65)
        else:
            full_response = response.json().get("response", "")

        return full_response

    except requests.exceptions.RequestException as e:
        print(f"\n[Error] Failed to connect to Ollama generation endpoint: {e}")
        raise e


if __name__ == "__main__":
    test_query = "What are the core characteristics of New Religious Movements in Islam?"
    try:
        generate_rag_answer(test_query, top_k=3)
    finally:
        close_qdrant_client()