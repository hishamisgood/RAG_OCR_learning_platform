"""Routing tests with scripted fake models - no Ollama, Chroma or network needed."""
import json

from langchain_core.documents import Document

from rag_graph.config import Settings
from rag_graph.graph import build_graph, run_question
from rag_graph.nodes import NO_ANSWER, Deps


class Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Answers based on the 'TASK: ...' tag on the first line of each prompt."""

    def __init__(self, relevant=True, grounded=("yes",)):
        self.relevant, self.grounded, self.tasks = relevant, list(grounded), []

    def invoke(self, prompt):
        task = prompt.splitlines()[0].replace("TASK: ", "")
        self.tasks.append(task)
        if task == "plan_query":
            return Msg("remote work policy")
        if task == "grade_relevance":
            return Msg(json.dumps({"relevant": "yes" if self.relevant else "no"}))
        if task == "generate":
            return Msg("Up to three days per week [1].")
        if task == "grade_groundedness":
            g = self.grounded.pop(0) if len(self.grounded) > 1 else self.grounded[0]
            return Msg(json.dumps({"grounded": g, "critique": "" if g == "yes" else "claim not in docs"}))
        raise AssertionError(task)


class FakeRetriever:
    def invoke(self, query):
        return [Document(page_content="Remote work: up to 3 days/week.", metadata={"source": "policy.md"})]


def make(llm, web=None):
    calls = []

    def web_search(q):
        calls.append(q)
        return web if web is not None else [Document(page_content="web fact", metadata={"source": "https://x.test", "title": "X"})]

    deps = Deps(llm=llm, grader_llm=llm, retriever=FakeRetriever(), web_search=web_search)
    return build_graph(deps, Settings(max_retries=2)), calls


def test_relevant_docs_skip_web_and_pass():
    llm = FakeLLM(relevant=True, grounded=["yes"])
    graph, web_calls = make(llm)
    out = run_question(graph, "How many remote days?")
    assert web_calls == [] and not out["used_web"]
    assert out["retries"] == 0 and "LOW CONFIDENCE" not in out["final_answer"]
    assert "policy.md" in out["final_answer"]


def test_irrelevant_docs_trigger_web_fallback():
    graph, web_calls = make(FakeLLM(relevant=False, grounded=["yes"]))
    out = run_question(graph, "Who won the 2022 World Cup?")
    assert web_calls == ["remote work policy"] and out["used_web"]
    assert "https://x.test" in out["final_answer"]


def test_no_evidence_abstains_without_looping():
    llm = FakeLLM(relevant=False)
    graph, _ = make(llm, web=[])
    out = run_question(graph, "Unanswerable?")
    assert out["generation"] == NO_ANSWER and out["retries"] == 0
    assert "generate" not in llm.tasks


def test_hallucination_loops_back_to_supervisor_then_passes():
    llm = FakeLLM(grounded=["no", "yes"])
    graph, _ = make(llm)
    out = run_question(graph, "How many remote days?")
    assert out["retries"] == 1 and out["grounded"] == "passed"
    assert llm.tasks.count("plan_query") == 2 and llm.tasks.count("generate") == 2


def test_retry_budget_is_bounded_and_flags_low_confidence():
    llm = FakeLLM(grounded=["no"])
    graph, _ = make(llm)
    out = run_question(graph, "How many remote days?")
    assert out["retries"] == 2 and llm.tasks.count("generate") == 3
    assert out["final_answer"].startswith("[LOW CONFIDENCE")


# ---- UI inputs must reach the nodes (they were silently dropped before GraphState declared them) ----

class RecordingRetriever:
    def __init__(self):
        self.subjects = "not called"

    def invoke(self, query, subjects=None):
        self.subjects = subjects
        return [Document(page_content="Remote work: up to 3 days/week.", metadata={"source": "policy.md"})]


def test_subject_filter_reaches_retriever():
    llm, retr = FakeLLM(relevant=True), RecordingRetriever()
    deps = Deps(llm=llm, grader_llm=llm, retriever=retr, web_search=lambda q: [])
    graph = build_graph(deps, Settings(max_retries=0))
    graph.invoke({"question": "remote days?", "subjects": ["Student"]}, config={"recursion_limit": 50})
    assert retr.subjects == ["Student"]


def test_allow_web_false_blocks_web_search():
    llm = FakeLLM(relevant=False)
    calls = []
    deps = Deps(llm=llm, grader_llm=llm, retriever=FakeRetriever(),
                web_search=lambda q: calls.append(q) or [])
    graph = build_graph(deps, Settings(max_retries=0))
    out = graph.invoke({"question": "remote days?", "allow_web": False}, config={"recursion_limit": 50})
    assert calls == [] and not out["used_web"]


def test_web_search_failure_abstains_instead_of_crashing():
    llm = FakeLLM(relevant=False)

    def unavailable(_query):
        raise TimeoutError("provider unavailable")

    deps = Deps(llm=llm, grader_llm=llm, retriever=FakeRetriever(), web_search=unavailable)
    graph = build_graph(deps, Settings(max_retries=0))
    out = run_question(graph, "Unanswerable?")
    assert out["generation"] == NO_ANSWER
    assert not out["used_web"]
    assert any("web_search: unavailable (TimeoutError)" in line for line in out["trace"])
