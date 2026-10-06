"""geoinv3d.io.filestore: inputs stored again are clones of the identical file stored before."""

import sys

import pytest

from geoinv3d.io import filestore


@pytest.fixture
def small_dedup(monkeypatch):
    monkeypatch.setattr(filestore, "MIN_DEDUP_BYTES", 10)


def test_an_identical_file_is_shared(tmp_path, small_dedup):
    index = tmp_path / "index.json"
    (tmp_path / "up1.tif").write_bytes(b"x" * 1000)
    (tmp_path / "up2.tif").write_bytes(b"x" * 1000)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert filestore.store(tmp_path / "up1.tif", tmp_path / "a" / "grid.tif", index) is None
    shared = filestore.store(tmp_path / "up2.tif", tmp_path / "b" / "grid.tif", index)
    assert shared == str(tmp_path / "a" / "grid.tif")
    assert (tmp_path / "b" / "grid.tif").read_bytes() == b"x" * 1000
    # the two can still be changed apart
    (tmp_path / "a" / "grid.tif").write_bytes(b"y" * 1000)
    assert (tmp_path / "b" / "grid.tif").read_bytes() == b"x" * 1000


def test_different_changed_or_small_files_are_not_shared(tmp_path, small_dedup):
    index = tmp_path / "index.json"
    (tmp_path / "one.tif").write_bytes(b"x" * 1000)
    (tmp_path / "other.tif").write_bytes(b"z" * 1000)
    filestore.store(tmp_path / "one.tif", tmp_path / "kept.tif", index)
    assert filestore.store(tmp_path / "other.tif", tmp_path / "kept2.tif", index) is None
    # the kept file changed since it was stored: its old hash no longer holds
    (tmp_path / "kept.tif").write_bytes(b"w" * 1000)
    assert filestore.store(tmp_path / "one.tif", tmp_path / "kept3.tif", index) is None
    (tmp_path / "tiny.csv").write_bytes(b"1,2")
    assert filestore.store(tmp_path / "tiny.csv", tmp_path / "tiny2.csv", index) is None
    assert (tmp_path / "tiny2.csv").read_bytes() == b"1,2"


def test_falls_back_to_a_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(filestore, "_clonefile", lambda src, dst: False)
    (tmp_path / "src.bin").write_bytes(b"abc" * 100)
    assert filestore.clone_or_copy(tmp_path / "src.bin", tmp_path / "dst.bin") is False
    assert (tmp_path / "dst.bin").read_bytes() == b"abc" * 100


@pytest.mark.skipif(sys.platform != "darwin", reason="APFS clones on macOS only")
def test_clones_on_apfs(tmp_path):
    (tmp_path / "src.bin").write_bytes(b"abc" * 100)
    (tmp_path / "dst.bin").write_bytes(b"old")
    assert filestore.clone_or_copy(tmp_path / "src.bin", tmp_path / "dst.bin") is True
    assert (tmp_path / "dst.bin").read_bytes() == b"abc" * 100
