"""One-off cleanup of data/docs.

- deletes unrelated / broken files and the docs.zip backup
- extracts every zip in data/docs into a sub-folder, then deletes the zip
- deletes byte-identical duplicate files (keeps the first, shortest path)

Usage:  python scripts/prepare_data.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "docs"

UNRELATED = ["code_of_conduct.md", "hardware_stipend.md", "remote_work_policy.md", "Unconfirmed 95317.crdownload"]
EXTRA = [ROOT / "data" / "docs.zip", ROOT / "rag_graph_learning_update"]


def _rm(p: Path, dry: bool) -> None:
    if not p.exists():
        return
    print(f"[delete] {p.relative_to(ROOT)}")
    if not dry:
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def _zip_name(info: zipfile.ZipInfo) -> str:
    """Zip files from Google Drive may store UTF-8 names without the UTF-8 flag."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def extract(z: Path, dry: bool) -> None:
    print(f"[extract] {z.name}")
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dst = DOCS / _zip_name(info)
            if DOCS.resolve() not in dst.resolve().parents:  # zip-slip guard
                print(f"  [skip unsafe] {info.filename}")
                continue
            print(f"  -> {dst.relative_to(DOCS)}")
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
    _rm(z, dry)


def dedupe(dry: bool) -> None:
    seen: dict[str, Path] = {}
    files = sorted((p for p in DOCS.rglob("*") if p.is_file() and p.stat().st_size), key=lambda p: (len(p.parts), str(p)))
    for p in files:
        h = hashlib.sha1(p.read_bytes()).hexdigest()
        if h in seen:
            print(f"[duplicate of {seen[h].relative_to(DOCS)}]", end=" ")
            _rm(p, dry)
        else:
            seen[h] = p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    for name in UNRELATED:
        _rm(DOCS / name, dry)
    for p in EXTRA:
        _rm(p, dry)
    for z in sorted(DOCS.glob("*.zip")):
        extract(z, dry)
    if not dry:
        dedupe(dry)
    print("done" + (" (dry run, nothing changed)" if dry else ""))


if __name__ == "__main__":
    main()
