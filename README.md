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
`up -d --pull always`, or an update tool). This repo only provides the script and the
service definition.

## Setup

1. Clone this repo next to the folder that holds your Immich `compose.yaml`, at the latest
   release tag. Run this, and the commands further down, from that Immich folder:

   ```bash
   git -C .. clone --branch vX.Y.Z https://github.com/mordilloSan/immich-ml-warmup.git
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
     extends:
       file: ../immich-ml-warmup/compose.yaml   # path to this repo's compose.yaml
       service: immich-ml-warmup
     image: *ml-image                           # the exact image immich-machine-learning runs
     user: "1000:1000"                          # same user as immich-machine-learning
     group_add: ["44", "991"]                   # this host's video and render group IDs
     networks: [immich-ml]                      # a network immich-machine-learning uses
     volumes:
       - ${MODEL_CACHE_LOCATION}:/cache         # same model cache as immich-machine-learning
     env_file: [.env]                           # same ML settings, plus the WARMUP_ variables
   ```

   Adjust each line to match your `immich-machine-learning` service: copy what it has, and
   leave out what it doesn't. Stock Immich, for example, runs ML as root on the default
   network, so its warm-up needs no `user`, `group_add` or `networks` line. The `/cache`
   volume and `env_file` are always needed. Each line you keep must match that service:

   - `user`: so both containers can write to the cache.
   - The `/cache` volume: a named volume (`model-cache:/cache`) works too.
   - `env_file`: the warm-up must see the same `MACHINE_LEARNING_` settings.
   - `group_add`: only with `user`. The groups that own `/dev/kfd` and
     `/dev/dri/renderD128` on your host (`stat -c '%g %n' /dev/kfd /dev/dri/renderD128`).
   - `networks`: the warm-up downloads models that aren't in the cache yet.

   You can add your own labels too, for example `io.linuxio.container.icon=si:amd`.

   Keep `image: *ml-image`: it is what keeps both services on the same image. Without it,
   the warm-up uses the repo's default line,
   `ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-rocm`, and nothing
   warns you if your ML line is different (a pinned tag, a digest). It would then compile
   for another MIGraphX build than the one `immich-machine-learning` runs.

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
| `warm-up: python exited with an error` | The script itself failed. See Known limits. |
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
- **This repo:** check out the new release tag, then recreate the container so it picks
  up the new script and settings:

  ```bash
  git -C ../immich-ml-warmup fetch --tags
  git -C ../immich-ml-warmup checkout vX.Y.Z
  docker compose up -d --force-recreate immich-ml-warmup
  ```

## Known limits

- While the warm-up compiles, a search or job that needs a model that isn't compiled yet
  makes `immich-machine-learning` compile the same model at the same time.
- The warm-up calls `preload_models` from `immich_ml.main`, which is internal Immich code. If
  a release renames it, the log says `warm-up: python exited with an error` and models
  compile during jobs again until this repo is updated.
- The healthcheck checks `/dev/dri/renderD128`, the first GPU. If yours is another render
  node, set `healthcheck` in your service.
- The container's filesystem is read-only. It writes only to `/cache` and to a 512 MB
  `/tmp`, which Docker mounts `noexec`. If a model compile ever fails with
  "Permission denied" on a `/tmp` path, give `/tmp` exec in your service. That entry replaces
  the repo's:

  ```yaml
  tmpfs:
    - /tmp:size=512m,mode=1777,exec
  ```

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
make check   # ruff, and docker compose config of compose.yaml
make test    # pytest
```

`make` uses `.venv` when it exists, and `python3` otherwise (CI installs the tools there).
Python 3.12 is the Python in the ROCm `immich-machine-learning` image. The tests don't need
the image, a GPU or a Docker daemon; `tests/test_compose.py` needs the `docker compose` CLI
and resolves `tests/stack/`, a minimal stack set up the way Setup describes.
