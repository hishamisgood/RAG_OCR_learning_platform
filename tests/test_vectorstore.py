from pathlib import Path

from rag_graph.config import Settings
from rag_graph.vectorstore import LoadResult


def test_ocr_disabled_marks_scanned_file_complete(monkeypatch, tmp_path):
    """Explicitly disabling OCR must not force the file to be re-ingested forever."""
    from rag_graph import vectorstore

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    scanned = docs_dir / "scan.pdf"
    scanned.write_bytes(b"placeholder")
    chroma_dir = tmp_path / "chroma"
    settings = Settings(docs_dir=docs_dir, chroma_dir=chroma_dir, ocr_provider="none")

    class Collection:
        def delete(self, **_kwargs):
            pass

    class Store:
        _collection = Collection()

        def delete_collection(self):
            pass

        def add_documents(self, _documents):
            pass

    result = LoadResult()
    result.failed_pages = 1
    monkeypatch.setattr(vectorstore, "_store", lambda *_args, **_kwargs: Store())
    monkeypatch.setattr(vectorstore, "load_file", lambda *_args, **_kwargs: result)

    vectorstore.ingest(settings, log=lambda _message: None)
    manifest = vectorstore._read_manifest(settings)
    assert manifest is not None
    assert "scan.pdf" in manifest
