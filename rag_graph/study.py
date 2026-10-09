"""Study tools that sit next to the graph (they reuse the same retriever and LLM)."""
from __future__ import annotations

from .llm import text_of
from .nodes import LEVELS, Deps, _context


def make_quiz(deps: Deps, topic: str, n: int = 5, level: str = "beginner",
              subjects: list[str] | None = None) -> str:
    """Retrieve material about `topic` and write n practice questions with answers."""
    docs = deps.retriever.invoke(topic, subjects=subjects) if subjects else deps.retriever.invoke(topic)
    if not docs:
        return "I could not find material about that topic in your books."
    prompt = (
        "TASK: quiz\n"
        f"You are a tutor. {LEVELS.get(level, LEVELS['beginner'])}\n"
        f"Write {n} practice questions about '{topic}' using ONLY the context below. "
        "After each question give a short answer and cite the source like [1].\n"
        "Format:\nQ1. <question>\nA1. <answer> [n]\n\n"
        f"Context:\n{_context(docs)}"
    )
    out = text_of(deps.llm.invoke(prompt))
    srcs = "\n".join(
        f"[{i}] {d.metadata.get('source', 'unknown')}" + (f" p.{d.metadata['page']}" if "page" in d.metadata else "")
        for i, d in enumerate(docs, 1)
    )
    return f"{out}\n\nSources:\n{srcs}"
