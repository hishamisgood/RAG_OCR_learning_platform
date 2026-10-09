"""Learning Platform - Streamlit UI for the CRAG + Self-RAG tutor.

Run:  streamlit run app.py
Tabs: Tutor Chat | Quiz Studio | Library & OCR Inspector
"""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import streamlit as st

from rag_graph.config import Settings
from rag_graph.graph import RUN_CONFIG, build_graph
from rag_graph.nodes import LEVELS

st.set_page_config(page_title="Learning Platform", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Palette: "Midnight Aurora"
#   night #0B1026 · surface #151B3B · teal #2DD4BF · violet #A78BFA
#   coral #FB7185 · amber #FBBF24 · text #E6E9F5 · muted #8B93B8
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
:root{--night:#0B1026;--surface:#151B3B;--surface2:#1E2650;--teal:#2DD4BF;--violet:#A78BFA;
      --coral:#FB7185;--amber:#FBBF24;--text:#E6E9F5;--muted:#8B93B8;}
html, body, [class*="css"], .stMarkdown, button, input, textarea {font-family:'Plus Jakarta Sans',sans-serif !important;}
.stApp{background:
   radial-gradient(1200px 600px at 10% -10%, rgba(167,139,250,.18), transparent 60%),
   radial-gradient(900px 500px at 100% 0%, rgba(45,212,191,.14), transparent 60%), var(--night);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#121838 0%,#0B1026 100%);border-right:1px solid #232B5A;}
.hero{padding:26px 30px;border-radius:20px;margin-bottom:18px;
      background:linear-gradient(120deg,rgba(45,212,191,.16),rgba(167,139,250,.18) 55%,rgba(251,113,133,.14));
      border:1px solid rgba(167,139,250,.35);}
.hero h1{margin:0;font-weight:800;font-size:2.3rem;
      background:linear-gradient(90deg,var(--teal),var(--violet) 55%,var(--coral));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{margin:.35rem 0 0;color:var(--muted);}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:var(--surface);padding:6px;border-radius:14px;border:1px solid #232B5A;}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:8px 18px;color:var(--muted);font-weight:600;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,var(--teal),var(--violet));color:#0B1026 !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
div[data-testid="stChatMessage"]{background:var(--surface);border:1px solid #232B5A;border-radius:16px;padding:14px 16px;}
.stButton>button,.stFormSubmitButton>button{background:linear-gradient(90deg,var(--teal),var(--violet));color:#0B1026;
      border:none;border-radius:12px;font-weight:700;transition:transform .15s, box-shadow .15s;}
.stButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-1px);box-shadow:0 6px 22px rgba(45,212,191,.35);color:#0B1026;}
.card{background:var(--surface);border:1px solid #232B5A;border-radius:16px;padding:16px 18px;}
.metric{background:var(--surface);border:1px solid #232B5A;border-radius:16px;padding:14px 18px;}
.metric .v{font-size:1.8rem;font-weight:800;color:var(--teal);} .metric .l{color:var(--muted);font-size:.85rem;}
.chip{display:inline-block;margin:4px 6px 0 0;padding:4px 10px;border-radius:999px;font-size:.8rem;
      border:1px solid #2B346B;background:var(--surface2);color:var(--text);text-decoration:none !important;}
.chip.notes{border-color:rgba(45,212,191,.55);} .chip.web{border-color:rgba(251,191,36,.6);}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.75rem;font-weight:700;margin-bottom:6px;}
.badge.notes{background:rgba(45,212,191,.15);color:var(--teal);} .badge.web{background:rgba(251,191,36,.15);color:var(--amber);}
.badge.low{background:rgba(251,113,133,.15);color:var(--coral);}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;}
.ocrtext{background:var(--surface);border:1px solid #232B5A;border-radius:12px;padding:14px;white-space:pre-wrap;
      max-height:720px;overflow:auto;font-size:.9rem;}
div[data-testid="stExpander"]{background:var(--surface);border:1px solid #232B5A;border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend (cached once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and the vector store…")
def backend():
    from rag_graph.main import build_deps

    s = Settings.from_env()
    deps = build_deps(s)
    return s, deps, build_graph(deps, s)


@st.cache_data(ttl=60, show_spinner=False)
def library_rows() -> tuple[list[dict], str | None]:
    from rag_graph.vectorstore import library

    try:
        return library(backend()[0]), None
    except Exception as e:
        return [], f"Could not load the indexed library: {type(e).__name__}: {e}"


@st.cache_data(ttl=30, show_spinner=False)
def engine_status() -> dict:
    s = Settings.from_env()
    ollama_ok, models = False, []
    try:
        import requests

        r = requests.get(f"{s.ollama_base_url}/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
        ollama_ok = True
    except Exception:
        pass
    return {"gemini": bool(s.gemini_key), "ollama": ollama_ok, "models": models,
            "llm": s.llm_model, "local": s.local_llm_model, "ocr": s.local_ocr_model}


@st.cache_data(show_spinner=False)
def _file_hash(path: str, sig: str) -> str:
    from rag_graph.ocr import file_hash

    return file_hash(Path(path))


s, deps, graph = backend()
rows, library_error = library_rows()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Learning Platform")
    st.caption("Your books & lecture notes, with a tutor that checks its own answers.")
    level = st.radio("Explanation level", list(LEVELS), horizontal=False,
                     format_func=lambda x: {"beginner": "🌱 Beginner", "intermediate": "🌿 Intermediate",
                                            "expert": "🌳 Expert"}[x])
    all_subjects = sorted({r["subject"] for r in rows if r["subject"]})
    subjects = st.multiselect("Limit to subjects", all_subjects, placeholder="All subjects")
    allow_web = st.toggle("Web search fallback", value=True,
                          help="When your notes don't cover the question, search the web (Tavily/Serper).")

    st.divider()
    es = engine_status()

    def _dot(ok):
        return f"<span class='dot' style='background:{'#2DD4BF' if ok else '#FB7185'}'></span>"

    st.markdown(
        f"{_dot(es['gemini'])}Gemini <code>{es['llm']}</code><br>"
        f"{_dot(es['ollama'])}Ollama fallback <code>{es['local']}</code><br>"
        f"{_dot(es['ocr'] in es['models'])}Local OCR <code>{es['ocr']}</code>",
        unsafe_allow_html=True,
    )
    if st.button("🧹 Clear chat", width = 'stretch'):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    "<div class='hero'><h1>Learning Platform</h1>"
    "<p>Ask questions, generate practice quizzes and inspect your scanned notes — "
    "grounded in your library, with the web as a backup.</p></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def source_chips(docs) -> str:
    chips = []
    for i, d in enumerate(docs or [], 1):
        m = d.metadata if hasattr(d, "metadata") else d
        src = str(m.get("source", "unknown"))
        if m.get("origin") == "web" or src.startswith("http"):
            label = html.escape(m.get("title") or src)[:70]
            chips.append(f"<a class='chip web' href='{html.escape(src)}' target='_blank'>🌐 [{i}] {label}</a>")
        else:
            page = f" · p.{m['page']}" if m.get("page") else ""
            ocr = " · OCR" if m.get("ocr") else ""
            chips.append(f"<span class='chip notes'>📘 [{i}] {html.escape(src)}{page}{ocr}</span>")
    return "".join(chips)


def render_answer(msg: dict) -> None:
    badges = ""
    if msg.get("docs"):
        badges += ("<span class='badge web'>🌐 Includes web search</span> " if msg.get("used_web")
                   else "<span class='badge notes'>📘 From your notes</span> ")
    body = msg["content"]
    if body.startswith("[LOW CONFIDENCE"):
        badges += "<span class='badge low'>⚠ Low confidence</span>"
        body = body.split("\n", 1)[1] if "\n" in body else ""
    if badges:
        st.markdown(badges, unsafe_allow_html=True)
    body = body.split("\n\nSources:\n")[0]  # sources are shown as chips below
    if "\n\nAnalogy (not from the books): " in body:
        body, analogy = body.split("\n\nAnalogy (not from the books): ", 1)
        st.markdown(body)
        st.info(f"💡 **Analogy** (not from the books): {analogy}")
    else:
        st.markdown(body)
    if msg.get("docs"):
        st.markdown(source_chips(msg["docs"]), unsafe_allow_html=True)
    if msg.get("trace"):
        with st.expander(f"🔎 How this answer was built ({msg.get('secs', 0):.1f}s)"):
            st.code("\n".join(msg["trace"]), language=None)


def parse_quiz(text: str) -> tuple[list[tuple[str, str]], str]:
    body, _, sources = text.partition("\n\nSources:\n")
    pairs = re.findall(r"Q\d+[.:)]\s*(.+?)\s*\n\s*A\d+[.:)]\s*(.+?)(?=\n\s*Q\d+[.:)]|\Z)", body, re.S)
    return [(q.strip(), a.strip()) for q, a in pairs], sources.strip()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_quiz, tab_lib = st.tabs(["💬 Tutor Chat", "📝 Quiz Studio", "📚 Library & OCR Inspector"])

# ---- Tutor chat ------------------------------------------------------------
with tab_chat:
    st.session_state.setdefault("messages", [])
    if library_error:
        st.error(library_error)
    elif not rows:
        st.warning("The library is empty. Run `python -m rag_graph ingest` (see the Library tab).")
    if not st.session_state.messages:
        st.markdown("<div class='card'>👋 Try: <i>“Explain Thevenin's theorem”</i>, "
                    "<i>“What is a Karnaugh map?”</i> or <i>“How does gradient descent work?”</i></div>",
                    unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🧠"):
            render_answer(m) if m["role"] == "assistant" else st.markdown(m["content"])

    if prompt := st.chat_input("Ask your tutor anything…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🧠"):
            trace, docs, used_web, final, t0 = [], [], False, None, time.perf_counter()
            with st.status("Thinking…", expanded=False) as status:
                try:
                    inp = {"question": prompt, "level": level, "subjects": subjects, "allow_web": allow_web}
                    for update in graph.stream(inp, config=RUN_CONFIG, stream_mode="updates"):
                        for node, delta in update.items():
                            delta = delta or {}
                            status.update(label=f"Running **{node.replace('_', ' ')}**…")
                            for line in delta.get("trace", []):
                                trace.append(line)
                                status.write(line)
                            if "documents" in delta:
                                docs = delta["documents"]
                            if "used_web" in delta:
                                used_web = delta["used_web"]
                            if "final_answer" in delta:
                                final = delta["final_answer"]
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Error", state="error")
                    final = f"⚠️ Something went wrong: `{e}`"
            msg = {"role": "assistant", "content": final or "(no answer produced)", "trace": trace,
                   "docs": [{**d.metadata} for d in docs], "used_web": used_web,
                   "secs": time.perf_counter() - t0}
            render_answer(msg)
            st.session_state.messages.append(msg)

# ---- Quiz studio -------------------------------------------------------------
with tab_quiz:
    with st.form("quiz"):
        c1, c2 = st.columns([4, 1])
        topic = c1.text_input("Topic", placeholder="e.g. Logic gates, AC circuits, Decision trees")
        n = c2.number_input("Questions", 1, 15, 5)
        go = st.form_submit_button("✨ Generate quiz", use_container_width=True)
    if go and topic.strip():
        from rag_graph.study import make_quiz

        with st.spinner(f"Writing {n} {level} questions about “{topic}”…"):
            try:
                st.session_state.quiz_data = (topic, make_quiz(deps, topic, int(n), level, subjects or None))
            except Exception as e:
                st.error(f"Could not generate the quiz: {e}")
    elif go:
        st.warning("Enter a topic first.")

    if "quiz_data" in st.session_state:
        qtopic, qtext = st.session_state.quiz_data
        pairs, sources = parse_quiz(qtext)
        st.markdown(f"#### Practice quiz · {html.escape(qtopic)}")
        if not pairs:
            st.markdown(qtext)
        for i, (q, a) in enumerate(pairs, 1):
            st.markdown(f"<div class='card' style='margin-top:10px'><b>Q{i}.</b> {html.escape(q)}</div>",
                        unsafe_allow_html=True)
            with st.expander("👀 Reveal answer"):
                st.markdown(a)
        if sources and pairs:
            st.caption("Sources")
            st.code(sources, language=None)

# ---- Library & OCR inspector ---------------------------------------------------
with tab_lib:
    if library_error:
        st.error(library_error)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lab in [
        (c1, len(rows), "Indexed files"),
        (c2, sum(r["chunks"] for r in rows), "Chunks"),
        (c3, sum(r["pages"] for r in rows), "Pages"),
        (c4, sum(r["ocr_pages"] for r in rows), "OCR'd pages"),
    ]:
        col.markdown(f"<div class='metric'><div class='v'>{val:,}</div><div class='l'>{lab}</div></div>",
                     unsafe_allow_html=True)
    st.write("")

    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True, column_config={
            "file": "File", "subject": "Subject",
            "chunks": st.column_config.ProgressColumn("Chunks", format="%d", min_value=0,
                                                      max_value=max(r["chunks"] for r in rows)),
            "pages": "Pages", "ocr_pages": "OCR pages"})

    with st.expander("⚙️ Index new / changed files"):
        st.caption("Only new or modified files are processed; OCR results are cached, so pages are never "
                   "OCR'd twice. Large libraries are faster from the terminal: `python -m rag_graph ingest`.")
        if st.button("Run ingest now"):
            from rag_graph.vectorstore import ingest

            bar, log_box, lines = st.progress(0.0), st.empty(), []

            def _log(msg):
                lines.append(msg)
                log_box.code("\n".join(lines[-15:]), language=None)

            def _on_file(i, total, rel):
                bar.progress(i / max(total, 1), text=f"{i}/{total} · {rel}")

            try:
                added = ingest(s, log=_log, on_file=_on_file)
                st.success(f"Added {added} chunks.")
            except BaseException as e:  # ingest raises SystemExit when the folder is empty
                st.error(str(e))
            library_rows.clear()
            st.rerun()

    st.markdown("#### 🔍 OCR inspector")
    ocr_files = [r for r in rows if r["ocr_pages"]]
    if not ocr_files:
        st.info("No OCR'd pages yet. Scanned pages show up here after ingestion.")
    else:
        from rag_graph.ocr import OCR, load_image, render_pdf_page

        f1, f2 = st.columns([3, 1])
        rel = f1.selectbox("File", [r["file"] for r in ocr_files],
                           format_func=lambda x: f"{x}  ({next(r['ocr_pages'] for r in ocr_files if r['file'] == x)} OCR pages)")
        path = s.docs_dir / rel
        if path.exists():
            st_ = path.stat()
            fhash = _file_hash(str(path), f"{st_.st_size}:{int(st_.st_mtime)}")
            cache_dir = s.ocr_cache_dir / fhash
            pages = sorted(int(p.stem) for p in cache_dir.glob("*.json")) if cache_dir.exists() else []
            if pages:
                page = f2.selectbox("Page", pages)
                hit = OCR(s, None, None).cached(fhash, page) or {}
                left, right = st.columns(2)
                with left:
                    if path.suffix.lower() == ".pdf":
                        import fitz

                        with fitz.open(str(path)) as doc:
                            img = render_pdf_page(doc, page - 1)
                    else:
                        img = load_image(path)
                    st.image(img, caption=f"{path.name} · page {page}", use_container_width=True)
                with right:
                    st.markdown(f"<span class='badge notes'>engine: {html.escape(hit.get('engine', '?'))}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"<div class='ocrtext' dir='auto'>{html.escape(hit.get('text', ''))}</div>",
                                unsafe_allow_html=True)
            else:
                st.info("No cached OCR text for this file.")
        else:
            st.warning("File no longer exists on disk.")
