"""
Configuration file containing project constants, connection parameters,
and structural Table of Contents mapping for the book PDF.
"""

from pathlib import Path

#System & Path Configurations
PDF_PATH = "data/book.pdf"
QDRANT_STORAGE_PATH = "qdrant_data"
COLLECTION_NAME = "NRM in Islam"

#Ollama API & Model Parameters
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"

# Nomic Embedding Specific Task Prefixes
NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "

#Chunking & Search Configurations
# Using character counts for predictable embedding token sizes
CHUNK_SIZE_CHARS = 1000       # Target ~200-250 words / ~250 tokens
CHUNK_OVERLAP_CHARS = 150     # ~30-40 words overlap to preserve context boundaries
DEFAULT_TOP_K = 5             # Number of context chunks retrieved for RAG generation

#Table of Contents & Metadata Structure
# Format: (start_page, end_page, section_title, [subsections])
BOOK_STRUCTURE = [
    {
        "section_id": 0,
        "title": "Introduction",
        "start_page": 7,
        "end_page": 11,
        "subsections": [
            {"title": "Overview", "start_page": 7, "end_page": 7},
            {"title": "Structure", "start_page": 7, "end_page": 7},
            {"title": "Terminology", "start_page": 8, "end_page": 10},
            {"title": "Classification", "start_page": 10, "end_page": 11},
        ],
    },
    {
        "section_id": 1,
        "title": "Authority Structures and the Limits on Religious Innovation",
        "start_page": 11,
        "end_page": 24,
        "subsections": [
            {"title": "Overview", "start_page": 11, "end_page": 12},
            {
                "title": "The First Divisions: Sunni, Shi‘i, and Ibadi",
                "start_page": 12,
                "end_page": 15,
            },
            {"title": "Imams and Madhhabs", "start_page": 15, "end_page": 16},
            {"title": "Imam-led NRMs", "start_page": 16, "end_page": 18},
            {"title": "Kaysanites", "start_page": 18, "end_page": 18},
            {"title": "Zaydis and Houthis", "start_page": 18, "end_page": 19},
            {
                "title": "Fatimids and the Esoteric",
                "start_page": 20,
                "end_page": 20,
            },
            {
                "title": "The Assassins and the Agha Khan",
                "start_page": 20,
                "end_page": 21,
            },
            {"title": "The Bohras", "start_page": 21, "end_page": 22},
            {"title": "The Sufi Revolution", "start_page": 22, "end_page": 23},
            {"title": "Conclusion", "start_page": 23, "end_page": 24},
        ],
    },
    {
        "section_id": 2,
        "title": "The Middle Period",
        "start_page": 24,
        "end_page": 32,
        "subsections": [
            {"title": "Overview", "start_page": 24, "end_page": 24},
            {"title": "Sunni Restorationism", "start_page": 24, "end_page": 25},
            {"title": "The Almohads", "start_page": 25, "end_page": 26},
            {"title": "Ibn Taymiyya", "start_page": 26, "end_page": 28},
            {"title": "Hybrid NRMs", "start_page": 28, "end_page": 28},
            {"title": "Nusayris and Druze", "start_page": 28, "end_page": 29},
            {
                "title": "Qizilbash, Bektashis, and Alevis",
                "start_page": 29,
                "end_page": 31,
            },
            {"title": "Conclusion", "start_page": 31, "end_page": 32},
        ],
    },
    {
        "section_id": 3,
        "title": "Early Modernity",
        "start_page": 32,
        "end_page": 43,
        "subsections": [
            {"title": "Overview", "start_page": 32, "end_page": 33},
            {"title": "Peripheral Jihad", "start_page": 33, "end_page": 34},
            {"title": "Wahhabism", "start_page": 34, "end_page": 36},
            {"title": "Other Jihads", "start_page": 36, "end_page": 38},
            {"title": "Liberal NRMs", "start_page": 38, "end_page": 39},
            {
                "title": "Sir Syed Ahmad Khan and the Aligarh Movement",
                "start_page": 40,
                "end_page": 41,
            },
            {
                "title": "Muhammad Abduh and the Renaissance",
                "start_page": 41,
                "end_page": 43,
            },
            {"title": "Conclusion", "start_page": 43, "end_page": 43},
        ],
    },
    {
        "section_id": 4,
        "title": "The Mass Age",
        "start_page": 43,
        "end_page": 53,
        "subsections": [
            {"title": "Overview", "start_page": 43, "end_page": 44},
            {
                "title": "The Muslim Brotherhood in Egypt",
                "start_page": 44,
                "end_page": 47,
            },
            {"title": "Nurcus and the Gülen Movement", "start_page": 47, "end_page": 47},
            {"title": "The Nurcus", "start_page": 48, "end_page": 48},
            {"title": "The Gülen Movement", "start_page": 49, "end_page": 50},
            {
                "title": "The Nation of Islam in the United States",
                "start_page": 50,
                "end_page": 53,
            },
            {"title": "Conclusion", "start_page": 53, "end_page": 53},
        ],
    },
    {
        "section_id": 5,
        "title": "Sufism in the West",
        "start_page": 53,
        "end_page": 60,
        "subsections": [
            {"title": "Overview", "start_page": 53, "end_page": 54},
            {"title": "The Early Sufis", "start_page": 54, "end_page": 54},
            {"title": "Inayat Khan", "start_page": 54, "end_page": 56},
            {"title": "The Maryamiyya", "start_page": 56, "end_page": 57},
            {"title": "The New Age", "start_page": 57, "end_page": 58},
            {
                "title": "Idries Shah and The Tradition",
                "start_page": 58,
                "end_page": 59,
            },
            {
                "title": "Sufi Sam and the Later Inayatis",
                "start_page": 59,
                "end_page": 60,
            },
            {"title": "Conclusion", "start_page": 60, "end_page": 60},
        ],
    },
    {
        "section_id": 6,
        "title": "Salafis and Jihadis",
        "start_page": 61,
        "end_page": 68,
        "subsections": [
            {"title": "Overview", "start_page": 61, "end_page": 61},
            {"title": "Salafism", "start_page": 61, "end_page": 62},
            {"title": "The JSM and a New Mahdi", "start_page": 63, "end_page": 63},
            {"title": "Salafism in the West", "start_page": 63, "end_page": 64},
            {"title": "Global Jihad", "start_page": 64, "end_page": 64},
            {"title": "Al-Qaeda", "start_page": 64, "end_page": 66},
            {"title": "Islamic State", "start_page": 66, "end_page": 67},
            {"title": "Conclusion", "start_page": 67, "end_page": 68},
        ],
    },
    {
        "section_id": 7,
        "title": "Modern Hybrids",
        "start_page": 68,
        "end_page": 76,
        "subsections": [
            {"title": "Overview", "start_page": 68, "end_page": 68},
            {"title": "Nigeria", "start_page": 68, "end_page": 68},
            {"title": "NAFSAT", "start_page": 68, "end_page": 70},
            {"title": "Chrislam", "start_page": 70, "end_page": 71},
            {"title": "The West", "start_page": 71, "end_page": 72},
            {"title": "Maryam Mosque", "start_page": 72, "end_page": 74},
            {"title": "Queer NRMs", "start_page": 74, "end_page": 76},
            {"title": "Conclusion", "start_page": 76, "end_page": 76},
        ],
    },
    {
        "section_id": 8,
        "title": "Conclusion",
        "start_page": 77,
        "end_page": 79,
        "subsections": [
            {"title": "Overview", "start_page": 77, "end_page": 79},
        ],
    },
    {
        "section_id": 9,
        "title": "Glossary",
        "start_page": 80,
        "end_page": 81,
        "subsections": [
            {"title": "Overview", "start_page": 80, "end_page": 81},
        ],
    },
]