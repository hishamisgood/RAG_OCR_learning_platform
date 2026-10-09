"""CLI:  python -m rag_graph ingest [--rebuild] | ocr | ask "..." [-v] [--level L] | chat [-v] [--level L] | quiz "topic" | mermaid"""
from __future__ import annotations

import argparse
import sys
import time

from .config import Settings
from .graph import RUN_CONFIG, build_graph
from .nodes import LEVELS, Deps


def build_deps(s: Settings) -> Deps:
    from .llm import get_llm
    from .vectorstore import get_retriever
    from .websearch import make_web_search

    return Deps(
        llm=get_llm(s),
        grader_llm=get_llm(s, json_mode=True),
        retriever=get_retriever(s),
        web_search=make_web_search(s),
    )


def build_app(s: Settings):
    return build_graph(build_deps(s), s)


def ask(graph, question: str, verbose: bool, level: str = "beginner") -> None:
    final, t0 = None, time.perf_counter()
    for update in graph.stream({"question": question, "level": level}, config=RUN_CONFIG, stream_mode="updates"):
        for node, delta in update.items():
            delta = delta or {}
            now = time.perf_counter()
            if verbose:
                print(f"  -> {node}  ({now - t0:.1f}s)")
                for line in delta.get("trace", []):
                    print(f"       {line}")
            t0 = now
            if "final_answer" in delta:
                final = delta["final_answer"]
    print("\n" + (final or "(no answer produced)") + "\n")


def _level(value: str) -> str:
    value = value.lower()
    if value not in LEVELS:
        raise SystemExit(f"Unknown level {value!r}. Choose from: {', '.join(LEVELS)}")
    return value


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # avoid Windows console encoding errors
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="rag_graph", description="CRAG + Self-RAG learning assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="index data/docs into ChromaDB (only new/changed files)")
    p_ing.add_argument("--rebuild", action="store_true", help="wipe the index and re-embed everything")
    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_chat = sub.add_parser("chat", help="interactive loop (type /level expert to change level)")
    p_quiz = sub.add_parser("quiz", help="practice questions about a topic")
    p_quiz.add_argument("topic")
    p_quiz.add_argument("-n", type=int, default=5, help="number of questions")
    for p in (p_ask, p_chat):
        p.add_argument("-v", "--verbose", action="store_true", help="print the node-by-node execution path")
    for p in (p_ask, p_chat, p_quiz):
        p.add_argument("--level", choices=list(LEVELS), help="explanation level (default: EXPLAIN_LEVEL or beginner)")
    sub.add_parser("mermaid", help="print the graph as a Mermaid diagram")
    sub.add_parser("ocr", help="OCR scanned pages into the cache only (no embedding); ingest later reuses it")
    args = parser.parse_args()

    s = Settings.from_env()
    level = _level(getattr(args, "level", None) or s.level)
    if args.cmd == "ingest":
        from .vectorstore import ingest

        added = ingest(s, rebuild=args.rebuild)
        print(f"Indexed {added} new chunks into {s.chroma_dir}")
    elif args.cmd == "ocr":
        from .ocr import QuotaExhausted, make_ocr
        from .vectorstore import list_files, load_file

        ocr = make_ocr(s)
        files = [p for p in list_files(s) if p.stat().st_size and p.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}]
        for n, p in enumerate(files, 1):
            print(f"[{n}/{len(files)}] {p.relative_to(s.docs_dir).as_posix()}", flush=True)
            try:
                load_file(s, p, ocr)
            except QuotaExhausted as e:
                print(f"[stopped] {e}")
                break
            except Exception as e:
                print(f"[skip] {p.name}: {e}")
        print(f"OCR pages: {ocr.stats if ocr else 'OCR disabled'}")
    elif args.cmd == "mermaid":
        stub = Deps(None, None, None, lambda q: [])
        print(build_graph(stub, s).get_graph().draw_mermaid())
    elif args.cmd == "ask":
        ask(build_app(s), args.question, args.verbose, level)
    elif args.cmd == "quiz":
        from .study import make_quiz

        print("\n" + make_quiz(build_deps(s), args.topic, args.n, level) + "\n")
    elif args.cmd == "chat":
        graph = build_app(s)
        print(f"Level: {level}.  Commands: /level beginner|intermediate|expert   (empty line to quit)")
        while (q := input("\n> ").strip()):
            if q.startswith("/level"):
                new = q.split(maxsplit=1)[1].strip().lower() if " " in q else ""
                if new in LEVELS:
                    level = new
                    print(f"Level set to {level}.")
                else:
                    print(f"Choose from: {', '.join(LEVELS)}")
                continue
            ask(graph, q, args.verbose, level)
