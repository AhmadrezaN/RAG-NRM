# Local High-Performance RAG Pipeline
### *New Religious Movements in Islam — Fully Local Vector Search & Streamed LLM Reasoning*

> **My first RAG pipeline:** A fully local, privacy-focused system that combines page-range-aware PDF chunking, persistent Qdrant vector search, and real-time LLM reasoning streaming based on the book *"New Religious Movements in Islam"* by Mark Sedgwick.

---

## Executive Summary

This project demonstrates a production-grade, fully local RAG implementation designed for simple academic texts. By decoupling retrieval inspection (/chunks) from generation, enforcing strict metadata schemas (Chapter, Subsection, Page Ranges), and parsing raw model reasoning tokens live in the CLI, the pipeline delivers complete transparency, zero data leakage, and high citation fidelity without relying on cloud APIs.

---

## Assumptions & Design Principles

1. **100% Offline & Private Capability**: No external API calls (OpenAI, Anthropic, etc.). All vector embeddings and LLM generations run locally on the user's hardware via Ollama and embedded Qdrant.
2. **Structural Page-Bound Chunking**: Generic character-count splitters break mid-sentence and lose citation context. The ingestion layer assumes strict page boundaries and metadata tracking so every retrieved chunk carries explicit `start_page` and `end_page` attributes.
3. **Asymmetric Embedding Retrieval**: Search queries and context chunks require different semantic representations. Using `nomic-embed-text`, queries are prefixed with `search_query: ` and documents with `search_document: ` to maximize similarity matching accuracy.
4. **Reasoning Transparency over Black-Box Answers**: Reasoning models like `qwen3:4b` generate step-by-step internal thoughts before answering. The system explicitly separates and streams this thinking process (`<think>` tags or native `thinking` fields) so users can verify *how* the model reached its conclusion based on retrieved contexts.

---
## Deep Dive: Chunking Policy & Ingestion Strategy

Standard RAG implementations often rely on naive, character-count splitters (e.g., fixed 500-token windows with 50-token overlaps). In academic texts, naive splitting breaks logical arguments mid-sentence and strips away vital source metadata. Our pipeline uses a **structurally grounded, page-bound chunking strategy**:

### 1. Structural Page Boundaries Over Fixed Character Windows
* **PyMuPDF (`fitz`) Extraction**: The document is ingested page by page while cross-referencing the structural book outline defined in `config.py`.
* **Metadata Attachment**: Instead of returning raw ungrounded strings, every extracted chunk is permanently bound to a specific **Chapter** and **Subsection**, carrying explicit `start_page` and `end_page` attributes.

### 2. Header and Footer Noise Elimination
* Running headers, footers, page numbers, and publisher artifacts occur on nearly every page of an academic PDF.
* The cleaning module (`cleaner.py`) uses targeted text cleaning routines to strip out this structural layout noise prior to vector generation.
* **Impact**: Removing repetitive layout text prevents Qdrant from returning high similarity scores on meaningless running headers, ensuring vector proximity reflects actual semantic content.

### 3. Asymmetric Search Optimization
* Queries and document passages require distinct vector representations. Using `nomic-embed-text`:
  * Every document chunk in `embedder.py` is prefixed with `search_document: `.
  * Every user query in `retriever.py` is prefixed with `search_query: `.
* **Impact**: Explicit asymmetric tagging allows Nomic to map concise, intent-driven user questions directly onto dense academic passages in 768-dimensional vector space.

### 4. Citation Preservation in Prompt Formatting
Because every chunk retains its exact structural metadata, `rag_generator.py` injects context blocks into the LLM prompt with explicit boundaries:




## Project Architecture & File Structure

```text
.
├── config.py           # Core configs: Ollama endpoints, model selection, book page boundaries
├── cleaner.py          # PyMuPDF text extraction, noise filtering, and metadata chunking
├── embedder.py         # Batch vector generator via Ollama 'nomic-embed-text'
├── vector_store.py     # Local Qdrant collection builder & payload indexer ('./qdrant_data')
├── retriever.py        # Vector search manager with Singleton Qdrant connection handling
├── rag_generator.py    # Context prompt builder & token-by-token reasoning/response stream parser
└── main.py             # Interactive CLI entrypoint with debug inspection modes

```

### Data & Execution Flow

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION & STORAGE                                  │
│                                                                                   │
│  [ PDF Book ] ──> cleaner.py ──> embedder.py ──> vector_store.py ──> [ Qdrant DB ]│
│                 (PDF Parser)    (Nomic 768-d)    (Local Storage)    (Disk Storage)│
└───────────────────────────────────────────────────────────────────────────────────┘
                                                                            │
┌───────────────────────────────────────────────────────────────────────────┼───────┐
│                              QUERY & STREAMING                            │       │
│                                                                           ▼       │
│  [ CLI UI ] <── rag_generator.py <── [ Grounded Prompt ] <── retriever.py          │
│   main.py      (Stream Parser)       (Context + Citations)   (Top-K Similarity)   │
└───────────────────────────────────────────────────────────────────────────────────┘

```

---

## Tech Stack & Dependencies

* **Language & Runtime**: Python 3.10+
* **PDF Processing**: `PyMuPDF` (`fitz`)
* **Embedding Model**: `nomic-embed-text` (768 dimensions via Ollama)
* **Vector Database**: `qdrant-client` (Embedded Local Disk Mode)
* **Generation & Reasoning Model**: `qwen3:4b` (via Ollama REST API)
* **HTTP Communications**: `requests` (with `iter_lines()` for live chunk streaming)

---

## Setup & Installation

### 1. Prerequisites

Ensure **Ollama** is installed and running on your local machine (`http://localhost:11434`).

Pull the required local models via terminal:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b

```

### 2. Environment Setup

```bash
# Clone repository and enter project directory
git clone [https://github.com/AhmadrezaN/RAG-NRM.git](https://github.com/AhmadrezaN/RAG-NRM.git)
cd RAG-NRM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install pymupdf qdrant-client requests

```

### 3. Build Vector Database

Run the ingestion pipeline to parse the book, build vector embeddings, and store them locally in `./qdrant_data`:

```bash
python vector_store.py

```

### 4. Run Interactive System

```bash
python main.py

```

---

## Verified Test Execution Report

The following log demonstrates a real end-to-end execution of the system queried about the **Zaydis**, showcasing context retrieval, live reasoning parsing, and final citation-grounded response generation:

```text
$ python main.py
=================================================================
  NEW RELIGIOUS MOVEMENTS IN ISLAM — LOCAL RAG SYSTEM
=================================================================
  Commands:
   • Type your question directly to generate an answer.
   • Type '/chunks <question>' to inspect raw vector matches.
   • Type 'exit', 'quit', or 'q' to close the session.
=================================================================

Query > tell me about zaydis.

[1/2] Retrieving top 3 relevant chunks from Qdrant...

=================================================================
 RETRIEVED CONTEXT CHUNKS (3 Selected)
=================================================================

[CHUNK #1] (Similarity Score: 0.6808)
Location : Authority Structures and the Limits on Religious Innovation -> Zaydis and Houthis (Pages 18-19)
-----------------------------------------------------------------
Zaydis and Houthis The Zaydis (far left line on Figure 2) moved to Yemen – a mountainous land that has always been difficult to control and so is hospitable for minorities – in the 890s. They established a line of ruling imams that lasted, with some interruptions, until 1970...
-----------------------------------------------------------------

[CHUNK #2] (Similarity Score: 0.68)
Location : Authority Structures and the Limits on Religious Innovation -> Zaydis and Houthis (Pages 18-19)
...

[2/2] Generating answer using 'qwen3:4b'...
=================================================================
[LLM THINKING PROCESS]:

I need to answer the question "tell me about zaydis" based solely on the provided context blocks from the book "New Religious Movements in Islam". Let me analyze the context blocks carefully.

First, I should note that there are three context blocks provided, but they all seem to be about the same topic (Zaydis and Houthis)...

Let me extract all the relevant information about Zaydis from the context blocks:
1. Zaydis moved to Yemen in the 890s...
2. Yemen is described as "a mountainous land that has always been difficult to control..."
3. They established a line of ruling imams that lasted until 1970...
...
I'll structure the answer to be comprehensive but focused on what the context tells us about Zaydis.

=================================================================
[LLM RESPONSE]:

# Zaydis: Overview from "New Religious Movements in Islam"
Based on the provided context from "New Religious Movements in Islam" (pages 18-19), here is information about Zaydis:
Zaydis are a Shia Islamic sect that moved to Yemen in the 890s. They chose this location because Yemen is described as "a mountainous land that has always been difficult to control and so is hospitable for minorities."
The Zaydis established a line of ruling imams that lasted, with some interruptions, until 1970. This line was overthrown after a civil war between the last Zaidi imam and a group of socialist army officers (Hovden 2022).
Between the 890s and 1970, there were several succession disputes and splits within the Zaydi community, but none of these led to a lasting alternative line of imams (as these are not shown in Figure 2 of the text).
The Zaydis' political and religious structure fits with the original conception of the imamate at the time of Ali: The imam was to have both religious and political authority, following the model of the Prophet. This historical pattern of establishing a state and dynasty aligns with this original conception.
Since 1970, there has been no Zaydi imam, but there are still many Zaydis in Yemen (though more Sunnis remain). The Houthi tribe is Zaydi in origin and forms the basis of the Houthi movement that conquered much of western Yemen during the civil war that started in 2014 (Salmoni, Loidolt, and Wells 2010, 94-107).
The text also notes that there were shorter-lived Zaydi states elsewhere in the region besides Yemen.
**Source**: Authority Structures and the Limits on Religious Innovation (Chapters 16-18, pages 18-19 of "New Religious Movements in Islam")
=================================================================

```

---

## Key Takeaways from System Execution

1. **Zero Hallucination Tolerance**: The LLM's internal thinking process shows it deliberately analyzing and extracting facts strictly present in Context Blocks 1–3 before outputting text.
2. **Metadata Integrity**: Citations naturally reflect precise metadata embedded during text extraction (`Pages 18-19`, `Authority Structures...`).
