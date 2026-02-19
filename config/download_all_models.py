#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def list_model_scripts(import_models_dir: Path) -> List[Path]:
    scripts = [
        p for p in import_models_dir.glob("*.py")
        if p.is_file() and not p.name.startswith("_")
    ]
    scripts.sort(key=lambda p: p.name.lower())
    return scripts


def run_script(script_path: Path, stop_on_error: bool) -> int:
    cmd = [sys.executable, str(script_path)]
    print(">", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Failed: {script_path.name} (exit code {e.returncode})")
        if stop_on_error:
            return e.returncode
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="import_models", help="Folder that contains model import scripts")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop at first failing script")
    ap.add_argument("--only", nargs="*", default=None, help="Run only scripts whose filename contains any of these strings")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    import_models_dir = (repo_root / args.dir).resolve()

    if not import_models_dir.exists():
        print(f"Folder not found: {import_models_dir}")
        return 2

    scripts = list_model_scripts(import_models_dir)

    if args.only:
        needles = [s.lower() for s in args.only]
        scripts = [p for p in scripts if any(n in p.name.lower() for n in needles)]

    if not scripts:
        print("No model import scripts found.")
        return 0

    for script in scripts:
        rc = run_script(script, stop_on_error=args.stop_on_error)
        if rc != 0:
            return rc

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())