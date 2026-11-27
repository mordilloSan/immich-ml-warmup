# immich-ml-warmup: a repo the Immich stack extends

Date: 2026-11-27. Status: draft, waiting for approval.

## Why

The warm-up service works, but its compose entry is about 120 lines: a Python script in a
heredoc plus long comments. That makes the Immich stack's compose file hard to read, and the
script cannot be linted or tested. The goal is a short entry in the stack, with the script
and everything that is not host-specific living in this repo, versioned and tested.

What the warm-up does stays the same: at each start it compares a stamp file
(`.warmup-stamp.json` in the model cache) with the current ML settings, image and compiled
files. If nothing changed it exits at once; otherwise it runs Immich's `preload_models` so
every model and OCR shape is compiled before a search or job needs it, writes the stamp, and
the container idles.

## Goals

- The stack's entry is about 12 lines; the rest lives in this repo.
- The warm-up always runs the exact image `immich-machine-learning` runs, whatever command
  updates the stack (`pull` then `up`, `up --pull always`, an update tool).
- Same behavior as today: same log lines, same stamp file, same fingerprint.
- The script is linted and unit-tested in CI.

## Non-goals

- A published or locally built image (see Approaches).
- CUDA or OpenVINO variants. The script is ROCm-specific (MIGraphX `.mxr` files, `/dev/kfd`).
- Fixing the known limit: while the warm-up compiles, ML may compile the same model at the
  same time.
- Detecting at runtime whether ML and the warm-up run different images.

## Anti-goals

- Any setup where the warm-up can end up on a different image than ML without an error.
- A refactor that changes the fingerprint's input or order. That would be a silent behavior
  change and force a full run on every host.
- New privileges or exposure: no Docker socket, no added capabilities, no ports.

## Constraints

- Compiled `.mxr` files only work with the onnxruntime and MIGraphX build that compiled them.
- `preload_models` and `settings` are internal Immich code (`immich_ml.main`). If a release
  renames them, the warm-up logs FAILED and models compile during jobs again.
- The script runs on the ML image's Python. Tests in CI must not need `immich_ml`, a GPU or
  Docker.
- Host: Docker Engine 29.10.1 with the containerd image store, Compose v5.8.0.

## Approaches

| Approach | Outcome |
|---|---|
| Dockerfile built by compose: `FROM immich-machine-learning:${IMMICH_VERSION}-rocm`, `pull_policy: build` | Tested on the host on 2026-11-27 with stand-in images. It follows local tag moves and version bumps, and `docker compose pull` skips it cleanly. But `docker compose up -d --pull always` exits 0 without rebuilding, so the warm-up stays on the old base while ML moves on: a silent mismatch. Rejected. |
| Ready-built image on GHCR | Needs an image for every Immich release and every move of `release`, pulls the multi-GB ROCm base on each build, and any delay is the same mismatch. Rejected. |
| **Stock ML image, script mounted from this repo, compose `extends`** (chosen) | The warm-up uses the same `image:` line as ML, so no update path can split them, and there is nothing to build. Cost: it is not an image; this repo is cloned on the host. |

## Design

### Layout

```
compose.yaml          service definition the stack extends
warmup/warmup.py      the script, mounted read-only at /warmup
tests/test_warmup.py  pytest unit tests
tests/test_compose.py checks that a sample stack extending compose.yaml resolves correctly
tests/stack/          that sample stack
pyproject.toml        ruff and pytest settings
requirements-dev.txt  pinned ruff and pytest
Makefile              check: ruff + compose config; test: pytest
README.md             what it does, setup, operation, updates, limits
```

The `warmup/` directory is mounted rather than the repo, so the container sees only the
script.

### compose.yaml in this repo

Holds everything that is the same on every host: `image` (the same expression as ML,
`ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-rocm`),
`container_name`, `security_opt`, `cap_drop`, `mem_limit`, `pids_limit`, `devices`,
`./warmup:/warmup:ro`, `init`, `entrypoint`, `healthcheck` and `restart`, with short comments
that point to the README. The entrypoint keeps today's shell wrapper, so the Python process
exits and frees its memory before the container idles:

```
/bin/sh -c 'python /warmup/warmup.py || echo "warm-up: python exited with an error";
            echo "warm-up: idle"; exec sleep infinity'
```

It does not set `env_file`, `user`, `group_add`, `networks`, the cache volume or labels.
Those are host-specific, and a relative `env_file` here would point at this repo's folder.

### Entry in the stack's compose file

The stack marks its `immich-machine-learning` image line with the YAML anchor `&ml-image`,
and the warm-up entry, placed after it, uses the alias. The two lines can't differ, and
removing the anchor makes compose stop with an error. This was added after the final review:
a README sentence alone left a silent mismatch for stacks whose ML line differs from the
repo's default.

```yaml
immich-ml-warmup:
  extends:
    file: ../immich-ml-warmup/compose.yaml   # this repo, checked out at a release tag
    service: immich-ml-warmup
  image: *ml-image                           # immich-machine-learning's own image line
  user: "1000:1000"                          # same user as immich-machine-learning
  group_add: ["44", "991"]                   # video and render group IDs on this host
  networks: [immich-ml]
  volumes:
    - ${MODEL_CACHE_LOCATION}:/cache         # same cache as immich-machine-learning
  env_file: [.env]                           # same ML settings, plus the WARMUP_ variables
  labels:
    - "io.linuxio.container.icon=si:amd"
```

Checked with `docker compose config` on Compose v5.8.0:

- A relative bind mount in the repo file resolves against the repo's folder.
- `${IMMICH_VERSION}` in the repo file is filled in from the stack's `.env`.
- `group_add` and `volumes` merge with the repo file's values.
- An `env_file` in the repo file would resolve to the repo's own folder, so it stays in the
  stack.

### Script

The script keeps the same code, split into functions so tests can import it without
`immich_ml`:

- `fingerprint()`
- `why_run(cache, fp)`
- `apply_warmup_env()`
- `write_stamp(cache, fp)`
- `main()`, run under `if __name__ == "__main__"`

The exact signatures are in the implementation plan.

The cache path is read from the environment when it is needed, not when the module is
imported. `immich_ml` is imported inside `main()` only after the skip check, as today. The
fingerprint's input and order stay exactly the same, and so do the log lines. A failed
preload still exits 0 and writes no stamp.

Run from a file instead of stdin, Python searches `/warmup` for installed packages instead of
the working directory `/usr/src`. Neither folder holds package metadata: the ML Dockerfile
installs dependencies with `uv sync --no-install-project` into `/opt/venv` and copies only
`immich_ml/` and `healthcheck.py` into `/usr/src`. So the fingerprint is the same as today's,
and the first start after migrating skips.

### Checks and tests

- `make check`: `ruff check`, `ruff format --check`,
  `docker compose -f compose.yaml config -q`.
- `make test` (pytest):
  - `fingerprint()`:
    - It is stable.
    - It changes with `WARMUP_`, `MACHINE_LEARNING_`, `IMMICH_` and `HSA_` variables and
      ignores others.
    - For the same inputs it matches a copy of the original inline code, which guards the
      anti-goal.
  - `why_run()`: no stamp, unreadable stamp, fingerprint changed, compiled file missing,
    compiled file changed size, and all matching (which returns `None`).
  - `WARMUP_` mapping.
  - Stamp writing: atomic replace and contents.
  - `main()`:
    - The skip path does not import `immich_ml`.
    - With a fake `immich_ml.main`, the success path writes the stamp and the failure path
      writes none.
- CI: `check.yml` sets up Python (action pinned by SHA) and installs `requirements-dev.txt`.
  Run `actionlint` after the change. Dependabot gets a `pip` entry.

### Acceptance on the host

This needs `sudo`, so it is done by the owner.

1. Switch the stack to the new entry and run `docker compose up -d immich-ml-warmup`. The log
   says "nothing changed since the last successful warm-up, skipped".
2. Delete the stamp file and restart the container. The run ends with "stamp written (N
   compiled files)".
3. In `docker compose config` for the stack, the warm-up's image equals ML's.

## Left to the builder

- The CI Python version is settled: 3.12, the Python in the ROCm image (Ubuntu 24.04;
  Immich requires `>=3.12`).
- The ruff rule set: defaults unless something in the script needs otherwise.
- The README wording, based on the comments in today's compose entry.

Releases go through gh-release-flow, when asked.
