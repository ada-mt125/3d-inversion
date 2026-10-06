"""Storing a job's input files without storing the same bytes again.

Every job keeps copies of its inputs (``~/.geoinv3d/inputs/<task>/data`` for re-runs, and the
local backend's ``local/<task>/data``), so a survey grid submitted fifty times was fifty copies
on disk (Block 8's 240 MB TMI grid: 54 copies, 12.8 GB).  ``store`` puts a file in place as a
clone of an identical file stored before, found by its size and hash in a small index; on APFS
(macOS) a clone shares the other file's blocks and takes no new space, and either file can
still be changed or deleted on its own.  Where cloning is not possible (another file system, or
another volume) it falls back to an ordinary copy.  Files under ``MIN_DEDUP_BYTES`` are copied.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

MIN_DEDUP_BYTES = 1_000_000
INDEX = Path.home() / ".geoinv3d" / "content_index.json"     # or GEOINV3D_CONTENT_INDEX
_lock = threading.Lock()
_libc = None


def file_hash(path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clonefile(src, dst) -> bool:
    """An APFS clone of src at dst (which must not exist); False where that is not possible."""
    global _libc
    if sys.platform != "darwin":
        return False
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            _libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            _libc.clonefile.restype = ctypes.c_int
        return _libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0
    except (OSError, AttributeError):
        return False


def clone_or_copy(src, dst) -> bool:
    """Put a copy of src at dst (replacing it): a clone where possible.  True if cloned."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp-store")
    if tmp.exists():
        tmp.unlink()
    cloned = _clonefile(src, tmp)
    if not cloned:
        shutil.copyfile(src, tmp)
    shutil.copymode(src, tmp)
    os.replace(tmp, dst)
    return cloned


def _load_index(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def store(src, dst, index: Optional[Path] = None) -> Optional[str]:
    """Put src at dst, as a clone of an identical file stored before when there is one.

    Returns the path of the stored file dst now shares its bytes with, or None (a new file).
    """
    src, dst = Path(src), Path(dst)
    size = src.stat().st_size
    if size < MIN_DEDUP_BYTES:
        clone_or_copy(src, dst)
        return None
    index = Path(index or os.environ.get("GEOINV3D_CONTENT_INDEX") or INDEX)
    digest = file_hash(src)
    key = f"{digest}:{size}"
    with _lock:
        entries = _load_index(index)
        known = [p for p in entries.get(key, []) if Path(p).exists() and Path(p).stat().st_size == size
                 and Path(p).resolve() != dst.resolve()]
        source = None
        for p in known:            # a stored file may have been changed since: check before sharing it
            if file_hash(p) == digest:
                source = p
                break
        clone_or_copy(source or src, dst)
        entries[key] = [str(dst)] + [p for p in known if p != str(dst)][:20]
        index.parent.mkdir(parents=True, exist_ok=True)
        tmp = index.with_name(index.name + ".tmp")
        tmp.write_text(json.dumps(entries), encoding="utf-8")
        os.replace(tmp, index)
    return source
