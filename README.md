# immich-ml-warmup

Compiles Immich's machine-learning models for AMD GPUs ahead of time, so searches and jobs
don't wait for a compile.

With the ROCm image, `immich-machine-learning` compiles each model (and each OCR input
shape) with MIGraphX the first time a search or job needs it, which makes that search or job
slow. This companion container does that work up front, every time the ML settings or the
image change.

## What it does

The `immich-ml-warmup` container runs next to `immich-machine-learning` and works in the
background; `immich-machine-learning` doesn't wait for it. At each start:

1. It checks a stamp file, `.warmup-stamp.json`, in the model cache. If the ML settings, the
   image and the compiled files are the same as after the last successful warm-up, it does
   nothing. That takes milliseconds and doesn't touch the GPU.
2. Otherwise it runs Immich's own preload code with the same image, cache, GPU and settings
   as `immich-machine-learning`, plus the models in your `WARMUP_` variables. Every model and
   OCR shape that isn't compiled yet gets compiled now. When everything is already compiled,
   this takes about 13 s.
3. It writes a new stamp. The Python process then exits, which frees its memory, and the
   container idles.

It needs the ROCm image (`immich-machine-learning:<version>-rocm`) and an AMD GPU.

Nothing is built and no extra image is pulled. The warm-up's `image:` is a YAML alias of
your `immich-machine-learning` image line (see Setup), so both services run the same image
whatever command you use to update Immich (`docker compose pull` then `up -d`,
`up -d --pull always`, or an update tool). This repo provides the script and the service
to paste into your Immich `compose.yaml`.

## Setup

1. Download the script from the latest release tag into the folder that holds your Immich
   `compose.yaml`. Run this, and the commands further down, from that folder:

   ```bash
   curl -fsSLo ml-warmup.py https://raw.githubusercontent.com/mordilloSan/immich-ml-warmup/vX.Y.Z/warmup/warmup.py
   ```

2. In your Immich `compose.yaml`, mark the `immich-machine-learning` image line with the
   anchor `&ml-image`. Leave the rest of the line as it is:

   ```yaml
   immich-machine-learning:
     image: &ml-image ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-rocm
   ```

3. Add the warm-up service below `immich-machine-learning` (a YAML alias only works after
   its anchor):

   ```yaml
   immich-ml-warmup:
     image: *ml-image                         # the exact image immich-machine-learning runs
     container_name: immich_ml_warmup
     user: "1000:1000"                        # same user as immich-machine-learning
     group_add: ["44", "991"]                 # this host's video and render group IDs
     networks: [immich-ml]                    # a network immich-machine-learning uses
     env_file: [.env]                         # same ML settings, plus the WARMUP_ variables
     volumes:
       - ${MODEL_CACHE_LOCATION}:/cache       # same model cache as immich-machine-learning
       - ./ml-warmup.py:/ml-warmup.py:ro      # the script from step 1
     devices:
       - /dev/dri:/dev/dri
       - /dev/kfd:/dev/kfd
     security_opt:
       - no-new-privileges:true
     cap_drop:
       - ALL
     read_only: true                          # writes only to /cache and /tmp
     tmpfs:
       - /tmp:size=512m,mode=1777             # ROCm's compiler temp files
     mem_limit: 16g
     pids_limit: 512
     init: true                               # so docker stop ends the idle sleep at once
     # Python exits after the warm-up and frees its memory; the container idles.
     entrypoint:
       - /bin/sh
       - -c
       - >-
         python /ml-warmup.py || echo "warm-up: python exited with an error";
         echo "warm-up: idle"; exec sleep infinity
     # Replaces the image's healthcheck, which pings an ML server that never runs here.
     # Healthy = able to do the job (cache writable, GPU usable), not "models compiled".
     healthcheck:
       test: ["CMD-SHELL", "test -w /cache && test -r /dev/kfd && test -w /dev/kfd && test -r /dev/dri/renderD128 && test -w /dev/dri/renderD128"]
       interval: 60s
       timeout: 5s
       retries: 3
       start_period: 30s
       start_interval: 2s
     restart: unless-stopped
   ```

   The lines from `user` to the `/cache` volume describe your host. Adjust them to match
   your `immich-machine-learning` service: copy what it has, and leave out what it doesn't.
   Stock Immich, for example, runs ML as root on the default network, so its warm-up needs
   no `user`, `group_add` or `networks` line. The `/cache` volume and `env_file` are always
   needed. Each line you keep must match that service:

   - `user`: so both containers can write to the cache.
   - The `/cache` volume: a named volume (`model-cache:/cache`) works too.
   - `env_file`: the warm-up must see the same `MACHINE_LEARNING_` settings.
   - `group_add`: only with `user`. The groups that own `/dev/kfd` and
     `/dev/dri/renderD128` on your host (`stat -c '%g %n' /dev/kfd /dev/dri/renderD128`).
   - `networks`: the warm-up downloads models that aren't in the cache yet.

   You can add your own labels too, for example `io.linuxio.container.icon=si:amd`.

   Keep `image: *ml-image` instead of writing the image again: it is what keeps both
   services on the same image. A copied line drifts as soon as you change the ML one (a
   pinned tag, a digest), nothing warns you, and the warm-up then compiles for another
   MIGraphX build than the one `immich-machine-learning` runs.

4. Start it, then read the log:

   ```bash
   docker compose up -d immich-ml-warmup
   docker logs immich_ml_warmup
   ```

## `WARMUP_` variables

Any `WARMUP_<NAME>=value` in `.env` becomes `<NAME>=value` inside the warm-up only, and
overrides `<NAME>` there. Use it to compile models that `immich-machine-learning` itself
doesn't preload, for example:

```bash
WARMUP_MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL=ViT-B-32__openai
```

## Operation

`docker logs immich_ml_warmup` shows what happened at the last start:

| Log line | Meaning |
|---|---|
| `warm-up: nothing changed since the last successful warm-up, skipped` | Nothing to do. |
| `warm-up: running, <reason>` | A run started. The reason is one of: `no stamp from an earlier warm-up`, `the ML settings or the image changed`, `compiled file is missing: <path>`, `compiled file changed size: <path>`, `unreadable stamp (...)`, `check failed (...)`. |
| `warm-up: all models loaded or compiled in <N> s` | The run succeeded. |
| `warm-up: stamp written (<N> compiled files)` | The next start can skip. |
| `warm-up FAILED after <N> s: <error>` | Immich's preload raised an error. No stamp is written, so it runs again at the next start. |
| `warm-up: python exited with an error` | The script itself failed, or `ml-warmup.py` is missing. See Known limits. |
| `warm-up: idle` | The container is done and waits. |

- To force a run, delete `.warmup-stamp.json` in the model cache and run
  `docker restart immich_ml_warmup`.
- A run that starts with `check failed (...)` writes no stamp, even when it succeeds, so
  the warm-up runs again at every start until the check works. It means the fingerprint
  couldn't be computed, for example because a GPU library in the image can't be read.
- "Healthy" means the warm-up is able to do its job: the cache is writable and the GPU
  devices are usable. It doesn't mean the models are compiled; the log says that.

## Updating

- **Immich:** nothing to do. The next start of the warm-up sees the new image and runs.
- **This repo:** download the script from the new release tag, then recreate the container
  so it runs the new script:

  ```bash
  curl -fsSLo ml-warmup.py https://raw.githubusercontent.com/mordilloSan/immich-ml-warmup/vX.Y.Z/warmup/warmup.py
  docker compose up -d --force-recreate immich-ml-warmup
  ```

  If the release notes mention a change to the service, update your `immich-ml-warmup`
  entry from Setup step 3 as well.

## Known limits

- While the warm-up compiles, a search or job that needs a model that isn't compiled yet
  makes `immich-machine-learning` compile the same model at the same time.
- The warm-up calls `preload_models` from `immich_ml.main`, which is internal Immich code. If
  a release renames it, the log says `warm-up: python exited with an error` and models
  compile during jobs again until this repo is updated.
- If `ml-warmup.py` isn't in the Immich folder when the container starts, Docker creates an
  empty folder of that name in its place, and the log says
  `warm-up: python exited with an error`. Remove that folder (`sudo rmdir ml-warmup.py`, it
  belongs to root), download the script (Setup step 1) and recreate the container.
- The healthcheck checks `/dev/dri/renderD128`, the first GPU. If yours is another render
  node, change it in the healthcheck.
- The container's filesystem is read-only. It writes only to `/cache` and to a 512 MB
  `/tmp`, which Docker mounts `noexec` and which counts against `mem_limit` while it holds
  files. If a model compile ever fails with "Permission denied" on a `/tmp` path, add
  `exec` to the `/tmp` line:

  ```yaml
  tmpfs:
    - /tmp:size=512m,mode=1777,exec
  ```

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
make check   # ruff
make test    # pytest
```

`make` uses `.venv` when it exists, and `python3` otherwise (CI installs the tools there).
Python 3.12 is the Python in the ROCm `immich-machine-learning` image. The tests don't need
the image, a GPU or a Docker daemon; `tests/test_compose.py` needs the `docker compose` CLI.
It pastes the two YAML blocks from Setup steps 2 and 3 into a minimal stack and resolves
it, so the README's service is the one that is tested.
