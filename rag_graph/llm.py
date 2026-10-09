"""LLM factory + helpers for parsing small-model JSON output.

Default setup costs nothing: Gemini's free tier (key without a billing account) is the
primary model, and the local Ollama model takes over whenever Gemini is unavailable
(no key, no internet, or the free daily quota is used up).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from .config import Settings

# ---- Gemini availability ("circuit breaker") ---------------------------------
# After a quota / rate-limit error we stop calling Gemini for a while instead of
# paying a failed round-trip on every single call.
_COOLDOWN_RATE = 60          # per-minute limit hit -> retry Gemini after 1 min
_COOLDOWN_QUOTA = 30 * 60    # daily quota exhausted -> retry after 30 min
_gemini_blocked_until = 0.0


def gemini_available() -> bool:
    return time.time() >= _gemini_blocked_until


def block_gemini(err: Exception) -> None:
    """Pause Gemini after an error; longer pause when it looks like the daily quota."""
    global _gemini_blocked_until
    msg = str(err).lower()
    daily = "per day" in msg or "perday" in msg or "daily" in msg
    _gemini_blocked_until = time.time() + (_COOLDOWN_QUOTA if daily else _COOLDOWN_RATE)
    print(f"[gemini] unavailable ({type(err).__name__}); using local model for "
          f"{'30 min' if daily else '1 min'}")


class FallbackLLM:
    """`.invoke(prompt)` on the primary model, falling back to the secondary on any error."""

    def __init__(self, primary: Any, secondary: Any, primary_name: str, secondary_name: str):
        self.primary, self.secondary = primary, secondary
        self.names = (primary_name, secondary_name)
        self.last_used = ""

    def invoke(self, prompt: Any, **kw) -> Any:
        if gemini_available():
            try:
                out = self.primary.invoke(prompt, **kw)
                self.last_used = self.names[0]
                return out
            except Exception as e:  # quota, network, safety block... -> local model
                block_gemini(e)
        self.last_used = self.names[1]
        return self.secondary.invoke(prompt, **kw)


def _ollama(s: Settings, json_mode: bool):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=s.local_llm_model, base_url=s.ollama_base_url, temperature=0,
        format="json" if json_mode else "",
        num_ctx=8192,                              # was 4096 (Ollama default); stops silent truncation
        num_predict=400 if json_mode else None,    # graders only need a short JSON verdict
        keep_alive="30m",                          # keep the model loaded between questions
    )


def _gemini(s: Settings, json_mode: bool):
    from langchain_google_genai import ChatGoogleGenerativeAI

    kw: dict[str, Any] = dict(
        model=s.llm_model, google_api_key=s.gemini_key, temperature=0,
        max_retries=1,            # fail fast -> local fallback instead of long back-off
        timeout=60,
    )
    if json_mode:
        kw["response_mime_type"] = "application/json"
        kw["thinking_budget"] = 0  # graders don't need "thinking"; saves quota + latency
    return ChatGoogleGenerativeAI(**kw)


def get_llm(s: Settings, json_mode: bool = False):
    """Chat model chosen by LLM_PROVIDER. Swap this function to use any LangChain chat model."""
    if s.llm_provider == "gemini" and s.gemini_key:
        return FallbackLLM(_gemini(s, json_mode), _ollama(s, json_mode),
                           f"gemini:{s.llm_model}", f"ollama:{s.local_llm_model}")
    if s.llm_provider == "gemini":
        print("[warn] LLM_PROVIDER=gemini but GEMINI_API_KEY is not set - using the local Ollama model.")
    return _ollama(s, json_mode)


def text_of(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # Gemini may return a list of content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content).strip()


def parse_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating extra prose around it."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def is_yes(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "y", "true", "1"}
