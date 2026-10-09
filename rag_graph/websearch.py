"""Web-search fallback used when CRAG judges the retrieved documents irrelevant."""
from __future__ import annotations

import os
from typing import Callable

from langchain_core.documents import Document

from .config import Settings

WebSearch = Callable[[str], "list[Document]"]


def make_web_search(s: Settings) -> WebSearch:
    provider = s.web_provider
    if provider == "tavily" and os.getenv("TAVILY_API_KEY"):
        return _tavily(s)
    if provider == "serper" and os.getenv("SERPER_API_KEY"):
        return _serper(s)
    if provider in {"tavily", "serper"}:
        print(f"[warn] WEB_PROVIDER={provider} but no API key set - web fallback disabled.")
    return lambda query: []


def _tavily(s: Settings) -> WebSearch:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def search(query: str) -> list[Document]:
        res = client.search(query=query, max_results=s.web_max_results)
        return [
            Document(page_content=r.get("content", ""), metadata={"source": r.get("url", ""), "title": r.get("title", "")})
            for r in res.get("results", [])
        ]

    return search


def _serper(s: Settings) -> WebSearch:
    import requests

    def search(query: str) -> list[Document]:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
            json={"q": query, "num": s.web_max_results},
            timeout=20,
        )
        resp.raise_for_status()
        return [
            Document(page_content=r.get("snippet", ""), metadata={"source": r.get("link", ""), "title": r.get("title", "")})
            for r in resp.json().get("organic", [])
        ]

    return search
