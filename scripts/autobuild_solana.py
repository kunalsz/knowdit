#!/usr/bin/env python3
"""Prebuild a Rust / Solana / Anchor repository into SBF `.so` artifacts.

Detects `Anchor.toml` or Cargo workspace program crates (`cdylib` / Solana deps)
and compiles them into `<repo>/target/deploy/*.so` so `knowdit workflow streamloop`
can execute them inside `LiteSVM` offline without rebuilding SBF binaries mid-audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SolanaBuildResult:
    repo_root: str
    used_docker: bool
    command: list[str]
    exit_code: int
    so_artifacts: list[str]
    stderr_tail: str


def find_so_artifacts(repo_root: Path) -> list[str]:
    deploy_dir = repo_root / "target" / "deploy"
    if not deploy_dir.is_dir():
        return []
    return sorted(str(p.relative_to(repo_root)) for p in deploy_dir.glob("*.so"))


def build_solana_workspace(
    repo_root: Path,
    cargo_build_sbf: str | None,
    anchor_bin: str | None,
    docker_image: str,
    force_docker: bool,
    timeout_secs: int,
) -> SolanaBuildResult:
    repo_root = repo_root.resolve()
    has_anchor = (repo_root / "Anchor.toml").is_file()

    local_sbf = (
        cargo_build_sbf
        or shutil.which("cargo-build-sbf")
        or (
            str(Path.home() / ".local/share/solana/install/active_release/bin/cargo-build-sbf")
            if (Path.home() / ".local/share/solana/install/active_release/bin/cargo-build-sbf").is_file()
            else None
        )
    )
    local_anchor = anchor_bin or shutil.which("anchor")

    if not force_docker and (local_sbf or (has_anchor and local_anchor)):
        if has_anchor and local_anchor:
            cmd = [local_anchor, "build"]
        else:
            assert local_sbf is not None
            cmd = [local_sbf]
        used_docker = False
    else:
        if not shutil.which("docker"):
            raise RuntimeError(
                "Neither `cargo-build-sbf` nor `docker` is available on $PATH. "
                "Run `./setup.sh` (installs Agave `cargo-build-sbf` and builds `knowdit:solana`)."
            )
        img_check = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if img_check.returncode != 0:
            raise RuntimeError(
                f"Local Docker image `{docker_image}` is not built yet and `cargo-build-sbf` is not on $PATH. "
                f"Run `./setup.sh` or `docker build -f docker/Dockerfile.solana -t {docker_image} docker`."
            )
        uid = os.getuid()
        gid = os.getgid()
        cmd = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{uid}:{gid}",
            "-v",
            f"{repo_root}:{repo_root}",
            "-w",
            str(repo_root),
            "--entrypoint",
            "/opt/solana/active_release/bin/cargo-build-sbf",
            docker_image,
        ]
        used_docker = True

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_secs,
    )
    artifacts = find_so_artifacts(repo_root)
    return SolanaBuildResult(
        repo_root=str(repo_root),
        used_docker=used_docker,
        command=cmd,
        exit_code=proc.returncode,
        so_artifacts=artifacts,
        stderr_tail=proc.stderr[-4000:],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path, help="Path to Rust / Solana repository")
    parser.add_argument("--cargo-build-sbf", default=None, help="Explicit cargo-build-sbf binary")
    parser.add_argument("--anchor", default=None, help="Explicit anchor binary")
    parser.add_argument(
        "--docker-image",
        default="knowdit:solana",
        help="Local Docker image built from docker/Dockerfile.solana when host SBF toolchain is absent",
    )
    parser.add_argument("--force-docker", action="store_true", help="Always build inside Docker")
    parser.add_argument("--timeout-secs", type=int, default=900, help="Build timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary to stdout")
    args = parser.parse_args()

    result = build_solana_workspace(
        repo_root=args.repo_root,
        cargo_build_sbf=args.cargo_build_sbf,
        anchor_bin=args.anchor,
        docker_image=args.docker_image,
        force_docker=args.force_docker,
        timeout_secs=args.timeout_secs,
    )
    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(
            f"autobuild_solana: exit={result.exit_code} docker={result.used_docker} "
            f"artifacts={result.so_artifacts}"
        )
        if result.exit_code != 0:
            sys.stderr.write(result.stderr_tail + "\n")
    return 0 if result.exit_code == 0 and result.so_artifacts else 1


if __name__ == "__main__":
    raise SystemExit(main())
