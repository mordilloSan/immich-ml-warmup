"""Resolve tests/stack, a minimal Immich stack that extends compose.yaml, with Docker Compose.

`docker compose config` needs only the Compose CLI, not a running Docker daemon.
"""

import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STACK = REPO / "tests" / "stack"
ENTRYPOINT = [
    "/bin/sh",
    "-c",
    (
        'python /warmup/warmup.py || echo "warm-up: python exited with an error"; '
        'echo "warm-up: idle"; exec sleep infinity'
    ),
]

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="needs docker compose"
)


def resolve(vars_file=None, compose_file=STACK / "compose.yaml"):
    """Services as Compose resolves them; variables come from vars_file (or the project's
    .env), never from this process's environment."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("IMMICH_VERSION", "MODEL_CACHE_LOCATION")
    }
    args = ["docker", "compose", "-f", str(compose_file)]
    if vars_file:
        args += ["--env-file", str(STACK / vars_file)]
    out = subprocess.run(
        [*args, "config", "--format", "json"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)["services"]


def test_readme_setup_resolves(tmp_path):
    """The README's stack entry, laid out as its Setup says, runs ML's image, written
    differently from the repo's default line."""
    readme = (REPO / "README.md").read_text()
    entry = re.search(r"```yaml\n( +immich-ml-warmup:\n.*?)\n +```", readme, re.DOTALL)[
        1
    ]
    (tmp_path / "immich-ml-warmup").symlink_to(
        REPO
    )  # the clone, next to the stack folder
    stack = tmp_path / "immich-app"
    stack.mkdir()
    (stack / ".env").write_text(
        "MACHINE_LEARNING_TEST=1\nMODEL_CACHE_LOCATION=/srv/model-cache\n"
    )
    (stack / "compose.yaml").write_text(
        "services:\n"
        "  immich-machine-learning:\n"
        "    image: &ml-image ghcr.io/immich-app/immich-machine-learning:v9.9.8-rocm\n"
        "    networks: [immich-ml]\n"
        + textwrap.indent(textwrap.dedent(entry), "  ")
        + "\nnetworks:\n  immich-ml:\n"
    )
    warmup = resolve(compose_file=stack / "compose.yaml")["immich-ml-warmup"]
    assert warmup["image"] == "ghcr.io/immich-app/immich-machine-learning:v9.9.8-rocm"
    assert Path(volume(warmup, "/warmup")["source"]).resolve() == REPO / "warmup"
    assert warmup["environment"]["MACHINE_LEARNING_TEST"] == "1"


def volume(service, target):
    return next(v for v in service["volumes"] if v["target"] == target)


@pytest.mark.parametrize(
    ("vars_file", "tag"),
    [("pinned.vars", "v9.9.9-rocm"), ("release.vars", "release-rocm")],
)
def test_warmup_runs_the_same_image_as_ml(vars_file, tag):
    services = resolve(vars_file)
    expected = f"ghcr.io/immich-app/immich-machine-learning:{tag}"
    assert services["immich-ml-warmup"]["image"] == expected
    assert services["immich-machine-learning"]["image"] == expected


def test_script_folder_mounted_read_only_from_repo():
    mount = volume(resolve("pinned.vars")["immich-ml-warmup"], "/warmup")
    assert mount["source"] == str(REPO / "warmup")
    assert mount["read_only"] is True


def test_stack_supplies_host_settings():
    warmup = resolve("pinned.vars")["immich-ml-warmup"]
    assert warmup["user"] == "1000:1000"
    assert warmup["group_add"] == ["44", "991"]
    assert volume(warmup, "/cache")["source"] == "/srv/model-cache"
    assert warmup["environment"]["MACHINE_LEARNING_TEST"] == "1"
    assert "immich-ml" in warmup["networks"]


def test_repo_file_keeps_hardening_and_entrypoint():
    warmup = resolve("pinned.vars")["immich-ml-warmup"]
    assert warmup["entrypoint"] == ENTRYPOINT
    assert warmup["init"] is True
    assert warmup["cap_drop"] == ["ALL"]
    assert warmup["security_opt"] == ["no-new-privileges:true"]
    assert warmup["pids_limit"] == 512
    assert warmup["container_name"] == "immich_ml_warmup"
    assert warmup["healthcheck"]["test"][0] == "CMD-SHELL"
    assert "ports" not in warmup
    assert "cap_add" not in warmup
    assert not warmup.get("privileged")
    assert not any("docker.sock" in v["source"] for v in warmup["volumes"])
