"""One function per box in the architecture diagram.

Every node takes the shared state and returns a partial state update.
Dependencies are injected (see `Deps`) so the whole graph can be tested
with fake models and swapped to other LLMs / vector stores / search APIs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from .config import Settings
from .llm import is_yes, parse_json, text_of
from .state import GraphState

NO_ANSWER = "I don't know - I could not find relevant information in the available sources."

# How the tutor should talk at each level.
LEVELS = {
    "beginner": (
        "Explain for a complete beginner. Use plain everyday words and short sentences, "
        "and define every technical term the first time you use it."
    ),
    "intermediate": (
        "Explain for someone who knows the basics. Use correct terminology, "
        "and walk through the reasoning step by step."
    ),
    "expert": (
        "Be concise and precise. Use technical terminology freely and focus on "
        "details, trade-offs and edge cases."
    ),
}

_ANALOGY_RE = re.compile(r"^\s*ANALOGY\s*:", re.I | re.M)
_EXPLAIN_RE = re.compile(r"^\s*EXPLANATION\s*:\s*", re.I)


def split_answer(text: str) -> tuple[str, str]:
    """Split a tutor reply into (grounded explanation, analogy). The analogy may be empty.

    Only the explanation is fact-checked: an analogy is *meant* to go beyond the books.
    """
    m = _ANALOGY_RE.search(text)
    grounded, extra = (text[: m.start()], text[m.end():]) if m else (text, "")
    return _EXPLAIN_RE.sub("", grounded.strip()).strip(), extra.strip()


@dataclass
class Deps:
    llm: Any                                    # chat model: .invoke(str) -> message
    grader_llm: Any                             # same model in JSON mode (for the evaluators)
    retriever: Any                              # .invoke(str) -> list[Document]
    web_search: Callable[[str], list[Document]]  # query -> list[Document]


def _context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def _pick(docs: list[Document], verdict: Any) -> list[Document]:
    """Keep the documents a batched relevance verdict points at.

    Accepts a list of 1-based numbers ([1, 3]) - the normal case - or a single yes/no
    (applies to all), which small local models sometimes return instead.
    """
    if isinstance(verdict, list):
        idx = set()
        for v in verdict:
            try:
                idx.add(int(str(v).strip(" []")))
            except ValueError:
                continue
        return [d for i, d in enumerate(docs, 1) if i in idx]
    return list(docs) if is_yes(verdict) else []


def make_nodes(deps: Deps, s: Settings) -> dict[str, Callable[[GraphState], dict]]:
    # 1. User Query ----------------------------------------------------------
    def user_query(state: GraphState) -> dict:
        q = state["question"].strip()
        return {
            "question": q, "documents": [], "generation": "", "grounded": "",
            "critique": "", "retries": 0, "used_web": False,
            "trace": [f"user_query: {q!r}"],
        }

    # 2. Supervisor Agent (planner / router) --------------------------------
    def supervisor(state: GraphState) -> dict:
        q, retries = state["question"], state.get("retries", 0)
        if state.get("grounded") == "hallucinated":  # loop-back from Self-RAG
            retries += 1
            prompt = (
                "TASK: plan_query\n"
                "A fact-checker rejected the previous answer.\n"
                f"Question: {q}\nPrevious answer: {state.get('generation', '')}\n"
                f"Critique: {state.get('critique', '')}\n"
                "Write ONE new search query likely to find the missing or correct evidence. "
                "Output only the query."
            )
        else:
            prompt = (
                "TASK: plan_query\n"
                f"Question: {q}\n"
                "Rewrite this as one concise, standalone search query. Output only the query."
            )
        lines = text_of(deps.llm.invoke(prompt)).strip('"\' ').splitlines()
        query = lines[0].strip('"\' ') if lines and lines[0].strip() else q
        return {"search_query": query, "retries": retries,
                "trace": [f"supervisor: query={query!r} (retry {retries}/{s.max_retries})"]}

    # 3. Retriever Agent ----------------------------------------------------
    def retriever(state: GraphState) -> dict:
        subjects = state.get("subjects") or None
        if subjects:
            docs = deps.retriever.invoke(state["search_query"], subjects=subjects)
        else:
            docs = deps.retriever.invoke(state["search_query"])
        scope = f" (subjects: {', '.join(subjects)})" if subjects else ""
        return {"documents": docs, "used_web": False,
                "trace": [f"retriever: {len(docs)} chunks{scope}"]}

    # 4. CRAG Evaluator (document relevance gate) ---------------------------
    # All chunks are graded in ONE call (instead of one call per chunk) to save free-tier quota.
    def crag_evaluator(state: GraphState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"documents": [], "relevance": "irrelevant",
                    "trace": ["crag_evaluator: 0 chunks -> irrelevant"]}
        prompt = (
            "TASK: grade_relevance\n"
            f"Question: {state['question']}\n\nDocuments:\n{_context(docs)}\n\n"
            "Which documents contain information useful for answering the question? "
            'Reply ONLY with JSON: {"relevant": [<numbers of the useful documents>]} '
            '(use an empty list if none are useful).'
        )
        verdict_raw = parse_json(text_of(deps.grader_llm.invoke(prompt))).get("relevant")
        kept = _pick(docs, verdict_raw)
        verdict = "relevant" if len(kept) >= s.min_relevant_docs else "irrelevant"
        return {"documents": kept, "relevance": verdict,
                "trace": [f"crag_evaluator: {len(kept)}/{len(docs)} relevant -> {verdict}"]}

    # 5. Web Search Fallback ------------------------------------------------
    def web_search(state: GraphState) -> dict:
        if state.get("allow_web") is False:
            return {"used_web": False, "trace": ["web_search: disabled by user"]}
        try:
            found = deps.web_search(state["search_query"])
        except Exception as e:
            # A fallback source must not turn a no-evidence answer into a failed request.
            # The generator will abstain when there is no usable local or web evidence.
            return {"used_web": False,
                    "trace": [f"web_search: unavailable ({type(e).__name__}); continuing without web results"]}
        for d in found:
            d.metadata.setdefault("origin", "web")
        return {"documents": state.get("documents", []) + found, "used_web": bool(found),
                "trace": [f"web_search: {len(found)} results"]}

    # 6. Generator Agent ----------------------------------------------------
    def generator(state: GraphState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {"generation": NO_ANSWER, "trace": ["generator: no evidence, abstaining"]}
        level = state.get("level") or s.level
        style = LEVELS.get(level, LEVELS["beginner"])
        prompt = (
            "TASK: generate\n"
            f"You are a patient tutor. {style}\n"
            "Use ONLY the context below for facts, and cite sources inline like [1] after each fact. "
            "If the context does not contain the answer, say you don't know.\n\n"
            "Format your reply exactly like this:\n"
            "EXPLANATION: <the explanation, every fact cited>\n"
            "ANALOGY: <one short everyday analogy or example to build intuition; "
            "it must not add new facts or numbers; write 'none' if it does not help>\n\n"
            f"Context:\n{_context(docs)}\n\nQuestion: {state['question']}\nAnswer:"
        )
        answer = text_of(deps.llm.invoke(prompt))
        return {"generation": answer, "trace": [f"generator: {len(answer)} chars"]}

    # 7. Self-RAG Evaluator (hallucination check) ---------------------------
    def selfrag_evaluator(state: GraphState) -> dict:
        docs = state.get("documents", [])
        if not docs:  # abstaining is trivially grounded
            return {"grounded": "passed", "critique": "", "trace": ["selfrag_evaluator: abstained -> passed"]}
        claims, _analogy = split_answer(state["generation"])  # the analogy is allowed to go beyond the books
        prompt = (
            "TASK: grade_groundedness\n"
            f"Documents:\n{_context(docs)}\n\nAnswer:\n{claims}\n\n"
            "Is every factual claim in the answer supported by the documents? "
            "Rewording or simplifying in plain language is fine as long as the meaning is preserved. "
            'Reply ONLY with JSON: {"grounded": "yes" or "no", "critique": "<unsupported claims, or empty>"}.'
        )
        data = parse_json(text_of(deps.grader_llm.invoke(prompt)))
        ok = is_yes(data.get("grounded"))  # unparsable output fails closed
        verdict = "passed" if ok else "hallucinated"
        critique = "" if ok else str(data.get("critique") or "grader flagged unsupported claims")
        return {"grounded": verdict, "critique": critique,
                "trace": [f"selfrag_evaluator: {verdict}" + (f" ({critique})" if critique else "")]}

    # 8. Final Output -------------------------------------------------------
    def output(state: GraphState) -> dict:
        claims, analogy = split_answer(state["generation"])
        answer = claims
        if analogy and analogy.strip(" .").lower() != "none":
            answer += "\n\nAnalogy (not from the books): " + analogy
        if state.get("grounded") != "passed":
            answer = "[LOW CONFIDENCE - could not be fully verified against the sources]\n" + answer
        docs = state.get("documents", [])
        if docs:
            lines = []
            for i, d in enumerate(docs, 1):
                title, src = d.metadata.get("title", ""), d.metadata.get("source", "unknown")
                page = f" p.{d.metadata['page']}" if "page" in d.metadata else ""
                lines.append(f"[{i}] {title + ' - ' if title else ''}{src}{page}")
            answer += "\n\nSources:\n" + "\n".join(lines)
        return {"final_answer": answer, "trace": ["output: done"]}

    return {
        "user_query": user_query, "supervisor": supervisor, "retriever": retriever,
        "crag_evaluator": crag_evaluator, "web_search": web_search, "generator": generator,
        "selfrag_evaluator": selfrag_evaluator, "output": output,
    }
