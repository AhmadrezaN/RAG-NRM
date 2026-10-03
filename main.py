"""
Interactive CLI Interface for "New Religious Movements in Islam" RAG System
Supports streaming live model reasoning/thinking processes and structured answers.
"""

import sys
from rag_generator import generate_rag_answer
from retriever import retrieve_relevant_chunks, close_qdrant_client


def display_banner():
    print("=" * 65)
    print("  NEW RELIGIOUS MOVEMENTS IN ISLAM — LOCAL RAG SYSTEM")
    print("=" * 65)
    print("  Commands:")
    print("   • Type your question directly to generate an answer.")
    print("   • Type '/chunks <question>' to inspect raw vector matches.")
    print("   • Type 'exit', 'quit', or 'q' to close the session.")
    print("=" * 65 + "\n")


def inspect_chunks_mode(query: str, top_k: int = 3):
    """Displays raw retrieved passages from Qdrant without calling the LLM."""
    print(f"\n[Searching Qdrant for top {top_k} matches...]")
    matches = retrieve_relevant_chunks(query, top_k=top_k)

    if not matches:
        print("No matching context found above the relevance threshold.")
        return

    print("\n" + "=" * 65)
    print(f" RETRIEVED CONTEXT CHUNKS ({len(matches)} Matches)")
    print("=" * 65)
    for i, match in enumerate(matches, start=1):
        print(f"\n--- Match #{i} (Score: {match.get('score', 'N/A')}) ---")
        print(f"Location: {match['chapter']} -> {match['subsection']} (Pages {match['start_page']}-{match['end_page']})")
        print(f"Text:\n{match['content']}")
        print("-" * 65)


def run_interactive_session():
    display_banner()

    try:
        while True:
            try:
                user_input = input("\nQuery > ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nShutting down local RAG session. Goodbye!")
                    break

                # Mode 1: Inspect raw vector chunks without running model generation
                if user_input.startswith("/chunks"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1 and parts[1].strip():
                        inspect_chunks_mode(parts[1].strip(), top_k=3)
                    else:
                        print("Usage: /chunks <your question here>")
                    continue

                # Mode 2: Standard RAG Generation pipeline with thinking stream
                generate_rag_answer(user_input, top_k=3)

            except KeyboardInterrupt:
                print("\n\n[Session interrupted by user]")
                continue

    finally:
        # Guarantee Qdrant database locks are safely released upon exit
        close_qdrant_client()


if __name__ == "__main__":
    run_interactive_session()