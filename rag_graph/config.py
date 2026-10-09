"""Central settings, overridable through environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"  # gemini (free tier, falls back to ollama) | ollama
    llm_model: str = "gemini-2.5-flash"
    local_llm_model: str = "llama3.2"  # Ollama model used offline / when the Gemini quota runs out
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "BAAI/bge-m3"  # multilingual (Arabic + English); was all-MiniLM-L6-v2
    docs_dir: Path = ROOT / "data" / "docs"
    chroma_dir: Path = ROOT / "chroma_db"
    collection: str = "docs"
    chunk_size: int = 1800       # books need bigger chunks than short docs (was 800)
    chunk_overlap: int = 250
    top_k: int = 6               # more context per question (was 4)
    level: str = "beginner"      # default explanation level: beginner | intermediate | expert
    min_relevant_docs: int = 1   # CRAG gate: how many relevant chunks count as "relevant"
    max_retries: int = 2         # Self-RAG loop-backs to the Supervisor before giving up
    web_provider: str = "tavily"  # tavily | serper | none
    web_max_results: int = 3
    # OCR for scanned pages / images
    ocr_provider: str = "gemini"  # gemini (falls back to ollama) | ollama | none
    ocr_model: str = "gemini-2.5-flash"
    local_ocr_model: str = "qwen2.5vl:3b"
    ocr_min_chars: int = 50       # a PDF page with less extractable text than this gets OCR'd
    ocr_min_interval: float = 6.5  # seconds between Gemini OCR calls (free tier ~10 requests/min)
    ocr_local_fallback: bool = True  # daily quota used up -> continue with the local model (False = stop, resume tomorrow)
    ocr_cache_dir: Path = ROOT / "data" / "ocr_cache"

    @property
    def gemini_key(self) -> str:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        g = os.getenv
        return cls(
            llm_provider=g("LLM_PROVIDER", d.llm_provider).lower(),
            llm_model=g("LLM_MODEL", d.llm_model),
            local_llm_model=g("LOCAL_LLM_MODEL", d.local_llm_model),
            ollama_base_url=g("OLLAMA_BASE_URL", d.ollama_base_url),
            embedding_model=g("EMBEDDING_MODEL", d.embedding_model),
            chunk_size=int(g("CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(g("CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(g("TOP_K", d.top_k)),
            level=g("EXPLAIN_LEVEL", d.level).lower(),
            min_relevant_docs=int(g("MIN_RELEVANT_DOCS", d.min_relevant_docs)),
            max_retries=int(g("MAX_RETRIES", d.max_retries)),
            web_provider=g("WEB_PROVIDER", d.web_provider).lower(),
            web_max_results=int(g("WEB_MAX_RESULTS", d.web_max_results)),
            ocr_provider=g("OCR_PROVIDER", d.ocr_provider).lower(),
            ocr_model=g("OCR_MODEL", d.ocr_model),
            local_ocr_model=g("LOCAL_OCR_MODEL", d.local_ocr_model),
            ocr_min_chars=int(g("OCR_MIN_CHARS", d.ocr_min_chars)),
            ocr_min_interval=float(g("OCR_MIN_INTERVAL", d.ocr_min_interval)),
            ocr_local_fallback=g("OCR_LOCAL_FALLBACK", str(d.ocr_local_fallback)).lower() in {"1", "true", "yes"},
        )
