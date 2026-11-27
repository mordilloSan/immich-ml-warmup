"""Tests for warmup/warmup.py.

Fakes stand in for Immich's ML code, the GPU libraries and the model cache, so these run
without the ML image, a GPU or Docker.
"""

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import types
from types import SimpleNamespace

import pytest

import warmup

STAMP = ".warmup-stamp.json"
SKIPPED = "warm-up: nothing changed since the last successful warm-up, skipped\n"


def _original_fingerprint():
    # The inline script's fingerprint(), verbatim: the move must not change the fingerprint,
    # or every host runs a full warm-up once after updating.
    parts = sorted(
        f"{k}={v}"
        for k, v in os.environ.items()
        if k.startswith(("WARMUP_", "MACHINE_LEARNING_", "IMMICH_", "HSA_"))
    )
    parts += sorted(
        f"{d.name}=={d.version}" for d in importlib.metadata.distributions()
    )
    libs = glob.glob("/opt/rocm*/lib/libmigraphx*.so*")
    ort = importlib.util.find_spec("onnxruntime")
    if ort and ort.submodule_search_locations:
        libs += glob.glob(
            os.path.join(ort.submodule_search_locations[0], "capi", "*.so*")
        )
    for lib in sorted(libs):
        st = os.stat(lib)
        parts.append(f"{lib}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


@pytest.fixture(autouse=True)
def restore_environ():
    # apply_warmup_env() writes to os.environ directly, outside monkeypatch's reach.
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "cache"
    path.mkdir()
    monkeypatch.setenv("MACHINE_LEARNING_CACHE_FOLDER", str(path))
    return path


def write_stamp_file(cache, data):
    (cache / STAMP).write_text(data if isinstance(data, str) else json.dumps(data))


def install_fake_immich(monkeypatch, preload_models, seen_env=None):
    """Stand-in for immich_ml.main; records the environment when settings is first read."""
    package = types.ModuleType("immich_ml")
    main = types.ModuleType("immich_ml.main")
    main.preload_models = preload_models

    def module_getattr(name):
        if name == "settings":
            if seen_env is not None:
                seen_env.update(os.environ)
            return SimpleNamespace(preload="PRELOAD")
        raise AttributeError(name)

    main.__getattr__ = module_getattr
    monkeypatch.setitem(sys.modules, "immich_ml", package)
    monkeypatch.setitem(sys.modules, "immich_ml.main", main)


# fingerprint()


def test_fingerprint_is_stable_hex():
    fp = warmup.fingerprint()
    assert fp == warmup.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", fp)


@pytest.mark.parametrize("prefix", ["WARMUP_", "MACHINE_LEARNING_", "IMMICH_", "HSA_"])
def test_fingerprint_changes_with_prefixed_variable(monkeypatch, prefix):
    before = warmup.fingerprint()
    monkeypatch.setenv(f"{prefix}TEST_X", "1")
    assert warmup.fingerprint() != before


def test_fingerprint_ignores_other_variables(monkeypatch):
    before = warmup.fingerprint()
    monkeypatch.setenv("UNRELATED_TEST_X", "1")
    assert warmup.fingerprint() == before


def test_fingerprint_matches_original_code(monkeypatch, tmp_path):
    rocm_lib = tmp_path / "libmigraphx_c.so.3"
    rocm_lib.write_bytes(b"a")
    ort_dir = tmp_path / "onnxruntime"
    (ort_dir / "capi").mkdir(parents=True)
    ort_lib = ort_dir / "capi" / "libonnxruntime_providers_migraphx.so"
    ort_lib.write_bytes(b"bb")
    real_glob = glob.glob
    monkeypatch.setattr(
        glob, "glob", lambda p: [str(rocm_lib)] if "libmigraphx" in p else real_glob(p)
    )
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(ort_dir)])
            if name == "onnxruntime"
            else None
        ),
    )
    monkeypatch.setenv("MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL", "ViT-B-32__openai")
    monkeypatch.setenv(
        "WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__VISUAL", "ViT-B-32__openai"
    )

    fp = warmup.fingerprint()
    assert fp == _original_fingerprint()
    # Both fake libraries count, so the equality above is not two empty library lists.
    rocm_lib.write_bytes(b"aa")
    assert warmup.fingerprint() != fp
    fp = warmup.fingerprint()
    ort_lib.write_bytes(b"bbb")
    assert warmup.fingerprint() != fp


# why_run()


def test_why_run_without_stamp(cache):
    assert warmup.why_run(cache, "fp") == "no stamp from an earlier warm-up"


def test_why_run_unreadable_stamp(cache):
    write_stamp_file(cache, "not json")
    assert warmup.why_run(cache, "fp").startswith("unreadable stamp (JSONDecodeError(")


def test_why_run_stamp_without_files(cache):
    write_stamp_file(cache, {"fingerprint": "fp"})
    assert warmup.why_run(cache, "fp") == "unreadable stamp (KeyError('files'))"


def test_why_run_fingerprint_changed(cache):
    write_stamp_file(cache, {"fingerprint": "old", "files": {}})
    assert warmup.why_run(cache, "fp") == "the ML settings or the image changed"


def test_why_run_compiled_file_missing(cache):
    write_stamp_file(cache, {"fingerprint": "fp", "files": {str(cache / "a.mxr"): 3}})
    assert warmup.why_run(cache, "fp") == f"compiled file is missing: {cache / 'a.mxr'}"


def test_why_run_compiled_file_changed_size(cache):
    (cache / "a.mxr").write_bytes(b"ab")
    write_stamp_file(cache, {"fingerprint": "fp", "files": {str(cache / "a.mxr"): 3}})
    assert (
        warmup.why_run(cache, "fp") == f"compiled file changed size: {cache / 'a.mxr'}"
    )


def test_why_run_nothing_changed(cache):
    (cache / "a.mxr").write_bytes(b"abc")
    write_stamp_file(cache, {"fingerprint": "fp", "files": {str(cache / "a.mxr"): 3}})
    assert warmup.why_run(cache, "fp") is None


# apply_warmup_env()


def test_apply_warmup_env_maps_and_overrides(monkeypatch):
    monkeypatch.setenv("MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL", "from-env")
    monkeypatch.setenv(
        "WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL", "ViT-B-32__openai"
    )
    warmup.apply_warmup_env()
    assert os.environ["MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL"] == "ViT-B-32__openai"
    assert (
        os.environ["WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL"]
        == "ViT-B-32__openai"
    )


# write_stamp()


def test_write_stamp_lists_nested_mxr_only(cache):
    (cache / "clip").mkdir()
    (cache / "clip" / "model.mxr").write_bytes(b"abc")
    (cache / "ocr" / "x").mkdir(parents=True)
    (cache / "ocr" / "x" / "det.mxr").write_bytes(b"abcde")
    (cache / "notes.txt").write_text("not a compiled model")

    assert warmup.write_stamp(cache, "fp") == 2
    stamp_text = (cache / STAMP).read_text()
    assert json.loads(stamp_text) == {
        "fingerprint": "fp",
        "files": {
            str(cache / "clip" / "model.mxr"): 3,
            str(cache / "ocr" / "x" / "det.mxr"): 5,
        },
    }
    assert stamp_text == json.dumps(json.loads(stamp_text), indent=1)
    assert not (cache / ".warmup-stamp.tmp").exists()
    assert warmup.why_run(cache, "fp") is None


# main()


def test_main_skips_without_importing_immich(cache, monkeypatch, capsys):
    warmup.write_stamp(cache, warmup.fingerprint())
    monkeypatch.setitem(sys.modules, "immich_ml", None)
    monkeypatch.setitem(sys.modules, "immich_ml.main", None)
    warmup.main()
    assert capsys.readouterr().out == SKIPPED


def test_main_runs_and_writes_stamp(cache, monkeypatch, capsys):
    monkeypatch.setenv(
        "WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL", "ViT-B-32__openai"
    )
    calls, seen_env = [], {}

    async def preload(preload_data):
        calls.append(preload_data)
        (cache / "clip").mkdir()
        (cache / "clip" / "model.mxr").write_bytes(b"abc")

    install_fake_immich(monkeypatch, preload, seen_env)
    warmup.main()
    out = capsys.readouterr().out
    assert calls == ["PRELOAD"]
    assert seen_env["MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL"] == "ViT-B-32__openai"
    assert "warm-up: running, no stamp from an earlier warm-up\n" in out
    assert re.search(
        r"^warm-up: all models loaded or compiled in \d+\.\d s$", out, re.MULTILINE
    )
    assert "warm-up: stamp written (1 compiled files)\n" in out

    os.environ.pop(
        "MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL"
    )  # a new container starts fresh
    warmup.main()
    assert capsys.readouterr().out == SKIPPED


def test_main_failed_preload_writes_no_stamp(cache, monkeypatch, capsys):
    async def preload(_):
        raise RuntimeError("boom")

    install_fake_immich(monkeypatch, preload)
    warmup.main()
    out = capsys.readouterr().out
    assert re.search(
        r"^warm-up FAILED after \d+\.\d s: RuntimeError\('boom'\)$", out, re.MULTILINE
    )
    assert not (cache / STAMP).exists()


def test_main_check_failed_runs_without_stamp(cache, monkeypatch, capsys):
    def broken_fingerprint():
        raise OSError("lib")

    async def preload(_):
        (cache / "a.mxr").write_bytes(b"x")

    monkeypatch.setattr(warmup, "fingerprint", broken_fingerprint)
    install_fake_immich(monkeypatch, preload)
    warmup.main()
    out = capsys.readouterr().out
    assert "warm-up: running, check failed (OSError('lib'))\n" in out
    assert "warm-up: all models loaded or compiled in" in out
    assert "stamp written" not in out
    assert not (cache / STAMP).exists()


def test_main_raises_when_immich_api_is_missing(cache, monkeypatch):
    monkeypatch.setitem(sys.modules, "immich_ml", None)
    with pytest.raises(ImportError):
        warmup.main()
