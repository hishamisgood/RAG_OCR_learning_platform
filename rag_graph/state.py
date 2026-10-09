"""Shared graph state passed between all nodes."""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str            # the user's original question
    level: str               # "beginner" | "intermediate" | "expert" (explanation depth)
    subjects: list[str]      # optional: limit retrieval to these subjects (folders / books)
    allow_web: bool          # False = never use the web fallback
    search_query: str        # Supervisor's (re)written retrieval query
    documents: list[Document]
    relevance: str           # "relevant" | "irrelevant"   (CRAG verdict)
    used_web: bool           # True once the web fallback has run
    generation: str          # candidate answer from the Generator
    grounded: str            # "passed" | "hallucinated"    (Self-RAG verdict)
    critique: str            # Self-RAG explanation, fed back to the Supervisor
    retries: int             # number of loop-backs so far
    final_answer: str
    trace: Annotated[list[str], add]  # human-readable execution log
