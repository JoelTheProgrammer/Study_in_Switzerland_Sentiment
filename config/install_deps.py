#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List, Optional

BASE_DEPS: Dict[str, str] = {
    "transformers": "transformers",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "praw": "praw",
    "pytest": "pytest",
    "matplotlib": "matplotlib",
}

TORCH_INDEX_URLS: Dict[str, str] = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
}


def run(cmd: List[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_pip() -> None:
    try:
        run([sys.executable, "-m", "pip", "--version"])
    except Exception:
        run([sys.executable, "-m", "ensurepip", "--upgrade"])
        run([sys.executable, "-m", "pip", "--version"])


def is_installed(dist_name: str) -> bool:
    import importlib.metadata as md
    try:
        md.version(dist_name)
        return True
    except md.PackageNotFoundError:
        return False


def pip_install(specs: List[str], upgrade: bool, extra_args: Optional[List[str]] = None) -> None:
    if not specs:
        return
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(specs)
    run(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", choices=sorted(TORCH_INDEX_URLS.keys()), default=None)
    ap.add_argument("--upgrade", action="store_true")
    args = ap.parse_args()

    ensure_pip()

    base_specs = [
        pip_spec
        for dist_name, pip_spec in BASE_DEPS.items()
        if args.upgrade or not is_installed(dist_name)
    ]
    pip_install(base_specs, upgrade=args.upgrade)

    if args.torch and (args.upgrade or not is_installed("torch")):
        pip_install(["torch"], upgrade=args.upgrade, extra_args=["--index-url", TORCH_INDEX_URLS[args.torch]])

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())