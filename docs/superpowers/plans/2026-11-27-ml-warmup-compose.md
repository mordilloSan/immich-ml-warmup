# immich-ml-warmup Compose Extends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the warm-up script and the host-independent service settings out of the Immich
stack's compose file into this repo, so the stack's warm-up entry shrinks to about 12 lines,
and lint and test the script in CI.

**Architecture:** The stack's `immich-ml-warmup` service uses compose `extends` to pull in
this repo's `compose.yaml`. That file runs the stock ROCm ML image (the same `image:` line as
`immich-machine-learning`) and mounts `warmup/` read-only, so nothing is built. Today's inline
script (Appendix A) becomes `warmup/warmup.py`, split into functions with unchanged behavior.

**Tech Stack:** Python 3.12, stdlib only at run time; pytest 9.2.0 and ruff 0.17.6 for
development; Docker Compose `extends`; GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-27-ml-warmup-compose-design.md`

## Global Constraints

- The script runs on the ROCm ML image's Python, 3.12, and uses only the standard library.
- Behavior is unchanged from Appendix A:
  - log lines word for word;
  - stamp file name `.warmup-stamp.json`;
  - stamp JSON `{"fingerprint": ..., "files": {path: size}}` written with `indent=1` through
    the temporary file `.warmup-stamp.tmp` and `os.replace`;
  - a failed preload prints FAILED, writes no stamp and exits 0.
- `fingerprint()` builds its input exactly as Appendix A does, with the same prefixes, sources,
  order and separators. It calls `glob.glob` and `importlib.util.find_spec` through their
  modules (`import glob`, `import importlib.util`), as Appendix A does, so tests can patch them.
- Image line in the repo's `compose.yaml`:
  `ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-rocm`.
- The repo's `compose.yaml` never sets `env_file`, `user`, `group_add`, `networks`, the
  `/cache` volume or `labels`; those belong to the stack.
- No Docker socket, no added capabilities, no ports.
- Pins:
  - `ruff==0.17.6` and `pytest==9.2.0`;
  - `actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0`;
  - the existing `actions/checkout` pin stays.
- Run `actionlint` after any workflow change.
- Never read or print `.env` files. Test fixtures use the `.vars` suffix.
- Never commit or push unless the user asks. If asked, branch off `main` first
  (`feat/compose-extends`). Each task ends with a suggested commit message instead of a
  commit. Releases go only through gh-release-flow, and only when asked.

## Review Focus

1. A stamp from a hand edit or an older version that lacks `files` gives the reason
   `unreadable stamp (KeyError('files'))`, then a run that rewrites the stamp. Test in Task 1.
2. A GPU library that can't be read (`os.stat` raises, for example a broken symlink) gives the
   reason `check failed (...)`. A full run follows and writes no stamp, so it runs again at the
   next start. Test in Task 1.
3. A future Immich release that renames or removes `immich_ml.main.preload_models`: `main()`
   raises `ImportError`, Python exits non-zero, and the shell wrapper logs "python exited with
   an error" and keeps the container idle instead of crash-looping. Test in Task 1 (exception
   propagates) and Task 2 (exact entrypoint string).
4. A `WARMUP_` variable whose target is already set in `.env` (for example both
   `MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL` and `WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL`):
   the `WARMUP_` value wins inside the warm-up, and only there. Test in Task 1.
5. A stack where `IMMICH_VERSION` is unset: both services resolve to `...:release-rocm`.
   Test in Task 2.

## Prerequisite: local tools (needs the user's OK)

This machine has no `pip`, `pytest`, `ruff` or `uv`, and its `python3` is 3.14 without
`ensurepip`. Install `uv` 0.13.9 as a single binary in `~/.local/bin`, checked against its
published SHA-256. `uv` then provides a Python 3.12 virtual environment in `.venv/`:

```bash
d=$(mktemp -d)
gh release download 0.13.9 -R astral-sh/uv -D "$d" \
  -p 'uv-x86_64-unknown-linux-gnu.tar.gz' -p 'uv-x86_64-unknown-linux-gnu.tar.gz.sha256'
(cd "$d" && sha256sum -c uv-x86_64-unknown-linux-gnu.tar.gz.sha256)
tar -xzf "$d/uv-x86_64-unknown-linux-gnu.tar.gz" -C "$d"
install -m 755 "$d/uv-x86_64-unknown-linux-gnu/uv" ~/.local/bin/uv
uv --version   # expect: uv 0.13.9
```

Every `make` and pytest command below runs with `PYTHON=.venv/bin/python`. CI uses the
runner's Python, so `PYTHON` keeps its default `python3` there.

---

### Task 1: Script as a tested module

**Files:**
- Create: `warmup/warmup.py`
- Create: `tests/test_warmup.py`
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Modify: `Makefile` (the `check` and `test` recipes)

**Interfaces:**
- Produces: module `warmup` (importable through pytest's `pythonpath = ["warmup"]`). The
  container runs it as `python /warmup/warmup.py`.

  ```python
  STAMP_NAME = ".warmup-stamp.json"
  def cache_dir() -> Path                       # Path(os.environ.get("MACHINE_LEARNING_CACHE_FOLDER", "/cache")), read per call
  def fingerprint() -> str                      # Appendix A fingerprint(), unchanged
  def why_run(cache: Path, fp: str) -> str | None   # Appendix A why_run(), stamp at cache / STAMP_NAME
  def apply_warmup_env() -> None                # WARMUP_X=v -> X=v in os.environ
  def write_stamp(cache: Path, fp: str) -> int  # writes the stamp, returns the number of .mxr files
  def main() -> None                            # Appendix A's top-level flow, `return` where it raised SystemExit(0)
  ```

- [ ] **Step 1: Set up the tools**

  Write these files:

  - `requirements-dev.txt`: two lines, `pytest==9.2.0` and `ruff==0.17.6`.
  - `pyproject.toml`:
    - `[tool.ruff]` with `target-version = "py312"` and nothing else, so ruff's default
      rules apply.
    - `[tool.pytest.ini_options]` with `pythonpath = ["warmup"]` and `testpaths = ["tests"]`.
  - `.gitignore`: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`.

  In the `Makefile`, add `PYTHON ?= python3` and replace the two recipe bodies:

  - `check`: `$(PYTHON) -m ruff check .` and `$(PYTHON) -m ruff format --check .`
  - `test`: `$(PYTHON) -m pytest`

  Keep `help` and its `##` descriptions.

  Run:
  `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt`

  Expected: `.venv/bin/python --version` prints `Python 3.12.x`.

- [ ] **Step 2: Write the failing tests in `tests/test_warmup.py`**

  Fixtures:

  - An autouse fixture saves `dict(os.environ)` and restores it after each test, because
    `apply_warmup_env` writes to `os.environ` directly.
  - `cache`: `tmp_path / "cache"`, created, with `MACHINE_LEARNING_CACHE_FOLDER` set to it.
  - `install_fake_immich(monkeypatch, preload_models, seen_env=None)`:
    - Puts `types.ModuleType` objects into `sys.modules` under `immich_ml` and
      `immich_ml.main`.
    - `immich_ml.main.preload_models = preload_models`.
    - Gives `immich_ml.main` a module-level `__getattr__`. For `"settings"` it copies
      `os.environ` into `seen_env` (when given) and returns
      `SimpleNamespace(preload="PRELOAD")`. That shows the `WARMUP_` variables were applied
      before Immich read its settings.

  Paste `_original_fingerprint()` into the test file: Appendix A's `fingerprint()` body copied
  verbatim. The test file imports what it needs at the top, one module per line.

  ```python
  def test_fingerprint_is_stable_hex():
      fp = warmup.fingerprint()
      assert fp == warmup.fingerprint() and re.fullmatch(r"[0-9a-f]{64}", fp)


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
      # fake libmigraphx file + fake onnxruntime/capi/*.so; patch glob.glob for the
      # "libmigraphx" pattern and importlib.util.find_spec("onnxruntime") to
      # SimpleNamespace(submodule_search_locations=[str(ort_dir)]); set a
      # MACHINE_LEARNING_ and a WARMUP_ variable
      assert warmup.fingerprint() == _original_fingerprint()


  # why_run, stamp written by hand as JSON in cache / ".warmup-stamp.json"
  def test_why_run_without_stamp(cache):
      assert warmup.why_run(cache, "fp") == "no stamp from an earlier warm-up"


  def test_why_run_unreadable_stamp(cache):  # stamp text "not json"
      assert warmup.why_run(cache, "fp").startswith("unreadable stamp (JSONDecodeError(")


  def test_why_run_stamp_without_files(cache):  # {"fingerprint": "fp"}
      assert warmup.why_run(cache, "fp") == "unreadable stamp (KeyError('files'))"


  def test_why_run_fingerprint_changed(cache):  # {"fingerprint": "old", "files": {}}
      assert warmup.why_run(cache, "fp") == "the ML settings or the image changed"


  def test_why_run_compiled_file_missing(
      cache,
  ):  # files {str(cache / "a.mxr"): 3}, no file
      assert warmup.why_run(cache, "fp") == f"compiled file is missing: {cache / 'a.mxr'}"


  def test_why_run_compiled_file_changed_size(cache):  # a.mxr is 2 bytes, stamp says 3
      assert (
          warmup.why_run(cache, "fp") == f"compiled file changed size: {cache / 'a.mxr'}"
      )


  def test_why_run_nothing_changed(cache):  # a.mxr is 3 bytes, stamp says 3
      assert warmup.why_run(cache, "fp") is None


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


  def test_write_stamp_lists_nested_mxr_only(cache):
      # cache/clip/model.mxr (3 bytes), cache/ocr/x/det.mxr (5 bytes), cache/notes.txt
      assert warmup.write_stamp(cache, "fp") == 2
      stamp_text = (cache / ".warmup-stamp.json").read_text()
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


  def test_main_skips_without_importing_immich(cache, monkeypatch, capsys):
      warmup.write_stamp(cache, warmup.fingerprint())
      monkeypatch.setitem(sys.modules, "immich_ml", None)
      monkeypatch.setitem(sys.modules, "immich_ml.main", None)
      warmup.main()
      assert (
          capsys.readouterr().out
          == "warm-up: nothing changed since the last successful warm-up, skipped\n"
      )


  def test_main_runs_and_writes_stamp(cache, monkeypatch, capsys):
      # WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL=ViT-B-32__openai set before main();
      # fake preload records its argument and creates cache/clip/model.mxr (3 bytes)
      warmup.main()
      out = capsys.readouterr().out
      assert calls == ["PRELOAD"]
      assert seen_env["MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL"] == "ViT-B-32__openai"
      assert "warm-up: running, no stamp from an earlier warm-up\n" in out
      assert re.search(
          r"^warm-up: all models loaded or compiled in \d+\.\d s$", out, re.M
      )
      assert "warm-up: stamp written (1 compiled files)\n" in out
      os.environ.pop(
          "MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL"
      )  # a new container starts with a fresh environment
      warmup.main()
      assert capsys.readouterr().out.endswith("skipped\n")


  def test_main_failed_preload_writes_no_stamp(cache, monkeypatch, capsys):
      # fake preload raises RuntimeError("boom")
      warmup.main()  # returns, no exception
      out = capsys.readouterr().out
      assert re.search(
          r"^warm-up FAILED after \d+\.\d s: RuntimeError\('boom'\)$", out, re.M
      )
      assert not (cache / ".warmup-stamp.json").exists()


  def test_main_check_failed_runs_without_stamp(cache, monkeypatch, capsys):
      # monkeypatch warmup.fingerprint to raise OSError("lib"); fake preload succeeds
      warmup.main()
      out = capsys.readouterr().out
      assert "warm-up: running, check failed (OSError('lib'))\n" in out
      assert "stamp written" not in out and not (cache / ".warmup-stamp.json").exists()


  def test_main_raises_when_immich_api_is_missing(cache, monkeypatch):
      monkeypatch.setitem(sys.modules, "immich_ml", None)
      with pytest.raises(ImportError):
          warmup.main()
  ```

- [ ] **Step 3: Run the tests and watch them fail**

  Run: `.venv/bin/python -m pytest -q`

  Expected: collection error `ModuleNotFoundError: No module named 'warmup'`.

- [ ] **Step 4: Write `warmup/warmup.py`**

  Move Appendix A into the Interfaces functions:

  - Keep its `print(..., flush=True)` calls and its exact strings, and import the same
    modules, one per line (ruff E401).
  - `main()` does what the top level of Appendix A does, in the same order: fingerprint and
    `why_run` inside the same `try`, then `apply_warmup_env()`, then
    `from immich_ml.main import preload_models, settings` inside `main()`, then the timed
    preload. It calls `write_stamp` only when `fp` is truthy, then prints
    `warm-up: stamp written ({n} compiled files)`.
  - Add a short module docstring that points to the README.
  - End with `if __name__ == "__main__": main()`.

- [ ] **Step 5: Run the tests and checks and watch them pass**

  Run: `make test PYTHON=.venv/bin/python && make check PYTHON=.venv/bin/python`

  Expected: all tests pass, and ruff reports `All checks passed!` and `N files already formatted`.
  If the format check fails, run `.venv/bin/python -m ruff format .` once and re-run.

- [ ] **Step 6: Suggested commit message**

  `feat: extract the warm-up script into a tested module`

---

### Task 2: Service definition the stack extends

**Files:**
- Create: `compose.yaml`
- Create: `tests/stack/compose.yaml`
- Create: `tests/stack/ml.vars`
- Create: `tests/stack/pinned.vars`
- Create: `tests/stack/release.vars`
- Create: `tests/test_compose.py`
- Modify: `Makefile` (`check` gains a compose line)

**Interfaces:**
- Consumes: `warmup/warmup.py` from Task 1 (mounted as `/warmup`).
- Produces: service `immich-ml-warmup` in `compose.yaml`. The stack's entry in
  `tests/stack/compose.yaml` is the reference copy that Task 4's README shows.

- [ ] **Step 1: Write the stack fixture**

  `tests/stack/compose.yaml` holds the two services and the network:

  ```yaml
  # A minimal Immich stack that uses this repo the way the README says.
  services:
    immich-machine-learning:
      image: ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-rocm
      env_file: [ml.vars]
      networks: [immich-ml]
    immich-ml-warmup:
      extends:
        file: ../../compose.yaml
        service: immich-ml-warmup
      user: "1000:1000"
      group_add: ["44", "991"]
      networks: [immich-ml]
      volumes:
        - ${MODEL_CACHE_LOCATION}:/cache
      env_file: [ml.vars]
  networks:
    immich-ml:
  ```

  The three `.vars` files:

  | File | Contents |
  |---|---|
  | `ml.vars` | `MACHINE_LEARNING_TEST=1` |
  | `pinned.vars` | `IMMICH_VERSION=v9.9.9` and `MODEL_CACHE_LOCATION=/srv/model-cache` |
  | `release.vars` | `MODEL_CACHE_LOCATION=/srv/model-cache` |

- [ ] **Step 2: Write the failing tests in `tests/test_compose.py`**

  The helper `resolve(vars_file) -> dict` runs
  `docker compose -f tests/stack/compose.yaml --env-file tests/stack/<vars_file> config --format json`.
  It removes `IMMICH_VERSION` and `MODEL_CACHE_LOCATION` from the subprocess environment, uses
  `check=True`, and returns `["services"]`. The module is skipped when
  `shutil.which("docker")` is `None`.

  ```python
  ENTRYPOINT = [
      "/bin/sh",
      "-c",
      'python /warmup/warmup.py || echo "warm-up: python exited with an error"; echo "warm-up: idle"; exec sleep infinity',
  ]


  @pytest.mark.parametrize(
      ("vars_file", "tag"),
      [("pinned.vars", "v9.9.9-rocm"), ("release.vars", "release-rocm")],
  )
  def test_warmup_runs_the_same_image_as_ml(vars_file, tag):
      svc = resolve(vars_file)
      assert (
          svc["immich-ml-warmup"]["image"]
          == svc["immich-machine-learning"]["image"]
          == f"ghcr.io/immich-app/immich-machine-learning:{tag}"
      )


  def test_script_folder_mounted_read_only_from_repo():
      vol = next(
          v
          for v in resolve("pinned.vars")["immich-ml-warmup"]["volumes"]
          if v["target"] == "/warmup"
      )
      assert vol["source"] == str(REPO / "warmup") and vol["read_only"] is True


  def test_stack_supplies_host_settings():
      w = resolve("pinned.vars")["immich-ml-warmup"]
      assert w["user"] == "1000:1000" and w["group_add"] == ["44", "991"]
      assert {"source": "/srv/model-cache", "target": "/cache"}.items() <= next(
          v for v in w["volumes"] if v["target"] == "/cache"
      ).items()
      assert (
          w["environment"]["MACHINE_LEARNING_TEST"] == "1"
          and "immich-ml" in w["networks"]
      )


  def test_repo_file_keeps_hardening_and_entrypoint():
      w = resolve("pinned.vars")["immich-ml-warmup"]
      assert w["entrypoint"] == ENTRYPOINT and w["init"] is True
      assert w["cap_drop"] == ["ALL"] and w["security_opt"] == ["no-new-privileges:true"]
      assert w["pids_limit"] == 512 and w["container_name"] == "immich_ml_warmup"
      assert w["healthcheck"]["test"][0] == "CMD-SHELL" and "ports" not in w
      assert not any("docker.sock" in v["source"] for v in w["volumes"])
  ```

- [ ] **Step 3: Run the tests and watch them fail**

  Run: `.venv/bin/python -m pytest tests/test_compose.py -q`

  Expected: FAIL with a `CalledProcessError`, because compose can't find `../../compose.yaml`.

- [ ] **Step 4: Write `compose.yaml`**

  The values below are fixed by the spec and by today's service. Add one comment line above
  the service that points to the README for the setup and for which keys the stack must add.

  ```yaml
  services:
    immich-ml-warmup:
      image: ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-rocm
      container_name: immich_ml_warmup
      security_opt:
        - no-new-privileges:true
      cap_drop:
        - ALL
      mem_limit: 16g
      pids_limit: 512
      devices:
        - /dev/dri:/dev/dri
        - /dev/kfd:/dev/kfd
      volumes:
        - ./warmup:/warmup:ro
      # init: so `docker stop` ends the idle `sleep` at once instead of after the 10 s timeout
      init: true
      # The Python process exits after the warm-up and frees its memory; the container idles.
      entrypoint:
        - /bin/sh
        - -c
        - >-
          python /warmup/warmup.py || echo "warm-up: python exited with an error";
          echo "warm-up: idle"; exec sleep infinity
      # Replaces the image's own healthcheck, which pings an ML server that never runs here.
      # Healthy = able to do the job (cache writable, GPU devices usable), not "models compiled".
      healthcheck:
        test: ["CMD-SHELL", "test -w /cache && test -r /dev/kfd && test -w /dev/kfd && test -r /dev/dri/renderD128 && test -w /dev/dri/renderD128"]
        interval: 60s
        timeout: 5s
        retries: 3
        start_period: 30s
        start_interval: 2s
      restart: unless-stopped
  ```

  Add `docker compose -f compose.yaml config -q` as the last line of the `Makefile`'s `check`
  recipe.

- [ ] **Step 5: Run the tests and checks and watch them pass**

  Run: `make test PYTHON=.venv/bin/python && make check PYTHON=.venv/bin/python`

  Expected: all tests pass, ruff is clean, and the compose config exits 0 with no output.

- [ ] **Step 6: Suggested commit message**

  `feat: add the service definition an Immich stack extends`

---

### Task 3: CI runs the checks and tests

**Files:**
- Modify: `.github/workflows/check.yml` (steps between checkout and `make check`)
- Modify: `.github/dependabot.yml` (a second `updates` entry)

**Interfaces:**
- Consumes: `make check`, `make test` and `requirements-dev.txt` from Tasks 1 and 2.

- [ ] **Step 1: Add Python to the `check` job**

  After the checkout step, add:

  - `actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0` with
    `python-version: "3.12"`;
  - a step `Install lint and test tools` running
    `python -m pip install -r requirements-dev.txt`.

  Keep the job id `check`, add no job name, and leave the existing comments and steps as they
  are.

- [ ] **Step 2: Add a `pip` entry to Dependabot**

  Copy the shape of the existing `github-actions` entry: `directory: /`, weekly,
  `cooldown.default-days: 7`, and a single group `dev-tools` with `patterns: ["*"]`. Set
  `commit-message.prefix: build` and `include: scope`.

- [ ] **Step 3: Verify**

  Run: `actionlint && make check PYTHON=.venv/bin/python && make test PYTHON=.venv/bin/python`

  Expected: actionlint prints nothing and exits 0, and both make targets pass.

- [ ] **Step 4: Suggested commit message**

  `ci: run ruff, compose config and pytest in the check job`

---

### Task 4: README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the stack entry in `tests/stack/compose.yaml` (Task 2), the log lines (Task 1)
  and the Makefile targets.

- [ ] **Step 1: Write `README.md`**

  Use these sections. Base the wording on Appendix B (today's compose comments) and the spec.

  1. **What it does.**
     - It checks the stamp and skips when nothing changed.
     - Otherwise it runs Immich's own `preload_models` with the same image, cache, GPU and
       settings as `immich-machine-learning`, plus the `WARMUP_` models, so every model and
       OCR shape is compiled before a search or job needs it. That takes about 13 s when all
       are already compiled.
     - The Python process then exits, which frees its memory, and the container idles.
     - ROCm only.
  2. **Setup.**
     - Clone at a release tag next to the stack: `git clone --branch vX.Y.Z <repo URL>`.
     - Add the stack entry. Copy it from `tests/stack/compose.yaml` with
       `file: ../immich-ml-warmup/compose.yaml`, `env_file: [.env]` and the optional
       `labels: ["io.linuxio.container.icon=si:amd"]`.
     - Say which keys must match `immich-machine-learning`: `user`, the `/cache` volume,
       `env_file`, `group_add` (the host's video and render groups) and `networks`.
     - If the stack's ML `image:` is written differently from the repo's line, set the same
       `image:` in the warm-up entry.
  3. **`WARMUP_` variables.**
     - `WARMUP_<NAME>=value` becomes `<NAME>=value` inside the warm-up only, and overrides
       `<NAME>` there.
     - Example: `WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL=ViT-B-32__openai` compiles a
       model that the ML server itself doesn't preload.
  4. **Operation.**
     - Read the result with `docker logs immich_ml_warmup`, and quote the log lines.
     - To force a run, delete `.warmup-stamp.json` in the model cache and restart the
       container.
     - A failed run writes no stamp, so it runs again at the next start.
     - "Healthy" means the cache is writable and the GPU devices are usable, not that the
       models are compiled.
  5. **Updating.** Run `git fetch --tags && git checkout vX.Y.Z` in the clone, then
     `docker compose up -d --force-recreate immich-ml-warmup` in the stack.
  6. **Known limits.**
     - While the warm-up compiles, a search or job that needs a model not compiled yet makes
       `immich-machine-learning` compile the same model at the same time.
     - `preload_models` is internal Immich code. If a release renames it, the log says
       `warm-up: python exited with an error` and models compile during jobs again.
  7. **Development.** `make check` and `make test`, plus the `uv venv` setup line from this
     plan's prerequisite.

- [ ] **Step 2: Verify**

  - Every log line quoted in the README appears verbatim in `warmup/warmup.py`.
  - The stack entry matches `tests/stack/compose.yaml` except for `file:`, `env_file:` and the
    label.
  - Run `make check PYTHON=.venv/bin/python` and check that it passes.

- [ ] **Step 3: Suggested commit message**

  `docs: add README with setup, operation and limits`

---

### Task 5: Acceptance on the host (the owner runs it; it needs sudo)

Not code. Hand the owner these steps after Tasks 1 to 4 pass review:

1. Clone or check out this repo next to the stack, and replace the stack's warm-up service
   with the README entry.
2. Run `sudo docker compose config immich-ml-warmup immich-machine-learning | grep image:`.
   Both lines must be the same.
3. Run `sudo docker compose up -d immich-ml-warmup && sudo docker logs immich_ml_warmup`.
   Expect `warm-up: nothing changed since the last successful warm-up, skipped` (the stamp
   from the old inline setup is still valid), then `warm-up: idle`.
4. Delete `.warmup-stamp.json` in the model cache, then run
   `sudo docker restart immich_ml_warmup`. Expect `warm-up: running, no stamp from an earlier
   warm-up`, `warm-up: all models loaded or compiled in …`, `warm-up: stamp written (N compiled
   files)` and `warm-up: idle`.

---

## Appendix A: today's inline script (same code; whitespace reformatted by ruff)

```python
import asyncio, glob, hashlib, importlib.metadata, importlib.util, json, os, time
from pathlib import Path

cache = Path(os.environ.get("MACHINE_LEARNING_CACHE_FOLDER", "/cache"))
stamp_file = cache / ".warmup-stamp.json"


def fingerprint():
    # Everything that decides which compiled files Immich needs: the ML settings, the
    # installed Python packages, and the size and date of the GPU libraries in the image.
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


def why_run(fp):
    # None when nothing changed since the last successful warm-up, else the reason to run
    try:
        stamp = json.loads(stamp_file.read_text())
        if stamp["fingerprint"] != fp:
            return "the ML settings or the image changed"
        for path, size in stamp["files"].items():
            if not os.path.isfile(path):
                return f"compiled file is missing: {path}"
            if os.path.getsize(path) != size:
                return f"compiled file changed size: {path}"
    except FileNotFoundError:
        return "no stamp from an earlier warm-up"
    except Exception as e:
        return f"unreadable stamp ({e!r})"
    return None


try:
    fp = fingerprint()
    reason = why_run(fp)
except Exception as e:
    fp, reason = None, f"check failed ({e!r})"
if reason is None:
    print(
        "warm-up: nothing changed since the last successful warm-up, skipped",
        flush=True,
    )
    raise SystemExit(0)
print(f"warm-up: running, {reason}", flush=True)

# WARMUP_MACHINE_LEARNING_PRELOAD__... -> MACHINE_LEARNING_PRELOAD__... (before Immich reads its settings)
for key, value in list(os.environ.items()):
    if key.startswith("WARMUP_"):
        os.environ[key.removeprefix("WARMUP_")] = value
from immich_ml.main import preload_models, settings

t = time.time()
try:
    asyncio.run(preload_models(settings.preload))
except Exception as e:
    print(f"warm-up FAILED after {time.time() - t:.1f} s: {e!r}", flush=True)
    raise SystemExit(0)
print(f"warm-up: all models loaded or compiled in {time.time() - t:.1f} s", flush=True)
if fp:
    files = {str(p): p.stat().st_size for p in cache.rglob("*.mxr")}
    tmp = stamp_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "files": files}, indent=1))
    os.replace(tmp, stamp_file)
    print(f"warm-up: stamp written ({len(files)} compiled files)", flush=True)
```

It ran as `python - <<'PY' … PY || echo "warm-up: python exited with an error"`, followed by
`echo "warm-up: idle"` and `exec sleep infinity`.

## Appendix B: today's compose comments (source for the README)

> Stays running and works in the background; immich-machine-learning does not wait for it.
> At each start it first checks a stamp file (.warmup-stamp.json in the model cache). When the
> ML settings, the image and the compiled files are the same as after the last successful
> warm-up, it does nothing (milliseconds, no GPU start). Otherwise it runs Immich's own
> preload code with the same image, cache, GPU and settings as immich-machine-learning, plus
> the models in the WARMUP_ variables of .env: every model and OCR shape that is not compiled
> yet is compiled now, instead of during a search or job (~13 s when all are already
> compiled). Then the Python process exits, which frees its memory, and the container idles.
> Known limit: while it compiles, a search or job that needs a model that is not compiled yet
> makes immich-machine-learning compile the same model at the same time.
> Result: `docker logs immich_ml_warmup`. To force a run: delete the stamp file and restart
> this container. A failed run writes no stamp, so it runs again at the next start.
> preload_models is internal Immich code: if a release renames it, the log says FAILED and
> models compile during jobs again.
