"""OCR for scanned PDF pages and images.

Primary engine: Gemini (free tier) vision.  Fallback: a local Ollama vision model.
Every page result is cached on disk (data/ocr_cache/<file-hash>/<page>.json), so a page is
OCR'd only once - even across `ingest --rebuild` or when the free daily quota runs out
half-way and ingest is resumed the next day.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Callable

from .config import Settings

MAX_SIDE = 2000  # px; enough for small handwriting, keeps requests small

PROMPT = (
    "You are an OCR engine for university study material (engineering / computer science).\n"
    "Transcribe ALL text on this page exactly as written, in reading order. The page may be "
    "printed or handwritten, in English, Arabic or both - keep each language as it is (do not translate).\n"
    "- If the page is rotated, read it in the correct orientation.\n"
    "- Keep math, formulas, truth tables and code; write formulas in plain text (e.g. X = A'B + AB').\n"
    "- For multiple-choice questions keep the options; if an option is ticked/circled, append '(marked)'.\n"
    "- For each diagram / circuit / graph write one line: [Figure: short description of what it shows].\n"
    "- Ignore watermarks, stamps, page borders and phone numbers on covers.\n"
    "Output only the transcription, no commentary. If the page has no content, output: [blank page]"
)

Engine = Callable[[bytes, str], str]  # (image bytes, mime type) -> text


class QuotaExhausted(RuntimeError):
    """Raised when the free daily quota is used up and no fallback is allowed."""


# ---- page images ---------------------------------------------------------------

def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render_pdf_page(doc, index: int) -> bytes:
    """Render one page of an open PyMuPDF document to PNG, longest side <= MAX_SIDE."""
    import fitz

    page = doc[index]
    zoom = min(MAX_SIDE / max(page.rect.width, page.rect.height), 200 / 72)  # cap at ~200 dpi
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")


def load_image(path: Path) -> bytes:
    """Read a photo/scan, fix EXIF rotation, downscale, return JPEG bytes."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ---- engines ---------------------------------------------------------------------

def gemini_engine(s: Settings) -> Engine:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=s.gemini_key)

    def run(image: bytes, mime: str) -> str:
        resp = client.models.generate_content(
            model=s.ocr_model,
            contents=[types.Part.from_bytes(data=image, mime_type=mime), PROMPT],
            config=types.GenerateContentConfig(temperature=0),
        )
        return (resp.text or "").strip()

    return run


def ollama_engine(s: Settings) -> Engine:
    import ollama

    client = ollama.Client(host=s.ollama_base_url)

    def run(image: bytes, mime: str) -> str:
        resp = client.chat(
            model=s.local_ocr_model,
            messages=[{"role": "user", "content": PROMPT, "images": [image]}],
            options={"temperature": 0, "num_ctx": 8192},
            keep_alive="10m",
        )
        return resp["message"]["content"].strip()

    return run


def _is_daily_quota(err: Exception) -> bool:
    m = str(err).lower()
    return "perday" in m or "per day" in m or "daily" in m


def _is_rate_limit(err: Exception) -> bool:
    m = str(err).lower()
    return ("429" in m or "resource_exhausted" in m or "quota" in m
            or re.search(r"\brate[ _-]?limit", m) is not None)


def _is_network(err: Exception) -> bool:
    m = str(err).lower()
    return any(t in m for t in ("getaddrinfo", "unreachable", "timed out", "timeout",
                                "connection", "503", "502", "temporarily"))


# ---- service -----------------------------------------------------------------------

class OCR:
    """Cached OCR with Gemini -> Ollama fallback.

    On a per-minute rate limit it waits and retries Gemini (OCR is a one-off job, quality
    matters more than speed). On the daily quota it switches to the local model, or raises
    QuotaExhausted when `local_fallback` is off so ingest can be resumed tomorrow.
    """

    def __init__(self, s: Settings, primary: Engine | None, fallback: Engine | None,
                 primary_name: str = "gemini", fallback_name: str = "ollama",
                 min_interval: float = 0.0, local_fallback: bool = True):
        self.s, self.primary, self.fallback = s, primary, fallback
        self.names = (primary_name, fallback_name)
        self.min_interval, self.local_fallback = min_interval, local_fallback
        self._primary_dead = primary is None
        self._last_call = 0.0
        self.stats = {"cache": 0, primary_name: 0, fallback_name: 0}

    # cache ---------------------------------------------------------------
    def _cache_file(self, fhash: str, page: int) -> Path:
        return self.s.ocr_cache_dir / fhash / f"{page}.json"

    def cached(self, fhash: str, page: int) -> dict | None:
        f = self._cache_file(fhash, page)
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _save(self, fhash: str, page: int, text: str, engine: str) -> None:
        f = self._cache_file(fhash, page)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"text": text, "engine": engine}, ensure_ascii=False), encoding="utf-8")

    # main entry ----------------------------------------------------------
    def page_text(self, fhash: str, page: int, image_fn: Callable[[], tuple[bytes, str]]) -> str:
        """Text of one page; `image_fn` is only called (rendered) on a cache miss."""
        hit = self.cached(fhash, page)
        if hit is not None:
            self.stats["cache"] += 1
            return hit["text"]
        image, mime = image_fn()
        text, engine = self._run(image, mime)
        self._save(fhash, page, text, engine)
        self.stats[engine] += 1
        return text

    def _run(self, image: bytes, mime: str) -> tuple[str, str]:
        if not self._primary_dead:
            for attempt in range(4):
                wait = self.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()
                try:
                    return self.primary(image, mime), self.names[0]
                except Exception as e:
                    if _is_daily_quota(e):
                        print(f"    [ocr] {self.names[0]} daily free quota used up")
                        self._primary_dead = True
                        break
                    if _is_rate_limit(e) and attempt < 3:
                        print("    [ocr] rate limit - waiting 30 s")
                        time.sleep(30)
                        continue
                    if _is_network(e) and attempt < 3:
                        print("    [ocr] network problem - retrying in 5 s")
                        time.sleep(5)
                        continue
                    print(f"    [ocr] {self.names[0]} failed: {str(e)[:120]}")
                    break
            if self._primary_dead and not self.local_fallback:
                raise QuotaExhausted("Gemini free quota used up - run ingest again tomorrow to continue.")
        if self.fallback is None:
            raise RuntimeError("no OCR engine available")
        return self.fallback(image, mime), self.names[1]


def make_ocr(s: Settings) -> OCR | None:
    if s.ocr_provider == "none":
        return None
    local = ollama_engine(s)
    if s.ocr_provider == "gemini" and s.gemini_key:
        return OCR(s, gemini_engine(s), local, f"gemini:{s.ocr_model}", f"ollama:{s.local_ocr_model}",
                   min_interval=s.ocr_min_interval, local_fallback=s.ocr_local_fallback)
    if s.ocr_provider == "gemini":
        print("[warn] OCR_PROVIDER=gemini but GEMINI_API_KEY is not set - using the local OCR model.")
    return OCR(s, None, local, "gemini", f"ollama:{s.local_ocr_model}")