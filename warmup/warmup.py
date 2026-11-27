"""Immich ML warm-up: compile every model before a search or job needs it.

Runs at container start inside the immich-machine-learning image, with the same cache, GPU
and settings. See README.md for setup, operation and known limits.
"""

import asyncio
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path

STAMP_NAME = ".warmup-stamp.json"


def cache_dir() -> Path:
    return Path(os.environ.get("MACHINE_LEARNING_CACHE_FOLDER", "/cache"))


def fingerprint() -> str:
    # Everything that decides which compiled files Immich needs: the ML settings, the
    # installed Python packages, and the size and date of the GPU libraries in the image.
    # Changing what goes in here, or its order, makes every host run once after updating.
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


def why_run(cache: Path, fp: str) -> str | None:
    """None when nothing changed since the last successful warm-up, else the reason to run."""
    try:
        stamp = json.loads((cache / STAMP_NAME).read_text())
        if stamp["fingerprint"] != fp:
            return "the ML settings or the image changed"
        for path, size in stamp["files"].items():
            if not os.path.isfile(path):
                return f"compiled file is missing: {path}"
            if os.path.getsize(path) != size:
                return f"compiled file changed size: {path}"
    except FileNotFoundError:
        return "no stamp from an earlier warm-up"
    except Exception as e:  # noqa: BLE001 -- any failure becomes the reason to run
        return f"unreadable stamp ({e!r})"
    return None


def apply_warmup_env() -> None:
    """WARMUP_<NAME>=value becomes <NAME>=value, before Immich reads its settings."""
    for key, value in list(os.environ.items()):
        if key.startswith("WARMUP_"):
            os.environ[key.removeprefix("WARMUP_")] = value


def write_stamp(cache: Path, fp: str) -> int:
    """Record the fingerprint and every compiled file; return how many files it lists."""
    stamp_file = cache / STAMP_NAME
    files = {str(p): p.stat().st_size for p in cache.rglob("*.mxr")}
    tmp = stamp_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "files": files}, indent=1))
    os.replace(tmp, stamp_file)
    return len(files)


def main() -> None:
    cache = cache_dir()
    try:
        fp = fingerprint()
        reason = why_run(cache, fp)
    except Exception as e:  # noqa: BLE001 -- any failure becomes the reason to run
        fp, reason = None, f"check failed ({e!r})"
    if reason is None:
        print(
            "warm-up: nothing changed since the last successful warm-up, skipped",
            flush=True,
        )
        return
    print(f"warm-up: running, {reason}", flush=True)

    apply_warmup_env()
    # Internal Immich code: if a release renames it, this import fails and Python exits
    # with an error, which the container's entrypoint logs before it idles.
    from immich_ml.main import preload_models, settings

    t = time.time()
    try:
        asyncio.run(preload_models(settings.preload))
    except Exception as e:  # noqa: BLE001 -- a failed preload is logged, never a crash
        print(f"warm-up FAILED after {time.time() - t:.1f} s: {e!r}", flush=True)
        return
    print(
        f"warm-up: all models loaded or compiled in {time.time() - t:.1f} s", flush=True
    )
    if fp:
        count = write_stamp(cache, fp)
        print(f"warm-up: stamp written ({count} compiled files)", flush=True)


if __name__ == "__main__":
    main()
