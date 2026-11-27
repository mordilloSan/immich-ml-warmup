"""Resolve the README's Setup snippets, pasted into a minimal Immich stack, with Docker Compose.

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

README = Path(__file__).resolve().parent.parent / "README.md"
ENV = "MACHINE_LEARNING_TEST=1\nMODEL_CACHE_LOCATION=/srv/model-cache\n"
ENTRYPOINT = [
    "/bin/sh",
    "-c",
    (
        'python /ml-warmup.py || echo "warm-up: python exited with an error"; '
        'echo "warm-up: idle"; exec sleep infinity'
    ),
]

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="needs docker compose"
)


def readme_block(first_line):
    """The README's yaml block that starts with first_line, unindented."""
    for block in re.findall(r"```yaml\n(.*?)\n *```", README.read_text(), re.DOTALL):
        block = textwrap.dedent(block)
        if block.startswith(first_line):
            return block
    raise AssertionError(f"README.md has no yaml block that starts with {first_line}")


def resolve(stack, env=ENV, ml=None):
    """Services of an Immich stack in the folder `stack`, made of the README's
    immich-machine-learning line (or `ml`) and warm-up service, as Compose resolves them.
    Variables come from the stack's .env, never from this process's environment."""
    (stack / ".env").write_text(env)
    (stack / "compose.yaml").write_text(
        "services:\n"
        + textwrap.indent(ml or readme_block("immich-machine-learning:"), "  ")
        + "\n"
        + textwrap.indent(readme_block("immich-ml-warmup:"), "  ")
        + "\nnetworks:\n  immich-ml:\n"
    )
    out = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(stack / "compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        env={
            k: v
            for k, v in os.environ.items()
            if k not in ("IMMICH_VERSION", "MODEL_CACHE_LOCATION")
        },
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)["services"]


@pytest.fixture
def warmup(tmp_path):
    return resolve(tmp_path)["immich-ml-warmup"]


def volume(service, target):
    return next(v for v in service["volumes"] if v["target"] == target)


@pytest.mark.parametrize(
    ("version", "tag"),
    [("IMMICH_VERSION=v9.9.9\n", "v9.9.9-rocm"), ("", "release-rocm")],
)
def test_warmup_runs_the_same_image_as_ml(tmp_path, version, tag):
    services = resolve(tmp_path, ENV + version)
    expected = f"ghcr.io/immich-app/immich-machine-learning:{tag}"
    assert services["immich-ml-warmup"]["image"] == expected
    assert services["immich-machine-learning"]["image"] == expected


def test_warmup_follows_an_ml_line_unlike_the_readme(tmp_path):
    """A pinned ML line, not the README's, still reaches the warm-up: it has no copy."""
    pinned = "ghcr.io/immich-app/immich-machine-learning:v9.9.8-rocm"
    ml = f"immich-machine-learning:\n  image: &ml-image {pinned}\n"
    assert resolve(tmp_path, ml=ml)["immich-ml-warmup"]["image"] == pinned


def test_script_mounted_read_only_from_the_stack_folder(tmp_path, warmup):
    mount = volume(warmup, "/ml-warmup.py")
    assert Path(mount["source"]).resolve() == (tmp_path / "ml-warmup.py").resolve()
    assert mount["read_only"] is True


def test_host_settings_match_ml(warmup):
    assert warmup["user"] == "1000:1000"
    assert warmup["group_add"] == ["44", "991"]
    assert volume(warmup, "/cache")["source"] == "/srv/model-cache"
    assert warmup["environment"]["MACHINE_LEARNING_TEST"] == "1"
    assert "immich-ml" in warmup["networks"]


def test_keeps_hardening_and_entrypoint(warmup):
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
    assert warmup["read_only"] is True
    assert warmup["tmpfs"] == ["/tmp:size=512m,mode=1777"]
    assert not any("docker.sock" in v["source"] for v in warmup["volumes"])
