"""Run commands on a remote Modal GPU sandbox."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path


def _stream_lines(stream, destination):
    for line in stream:
        destination.write(line)
        destination.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command on a Modal GPU sandbox.")
    parser.add_argument("--gpu", default="T4", choices=["T4", "A10G", "A100", "H100"])
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in minutes")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Expected a command after `--`.")

    try:
        import modal
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"Failed to import modal: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        .add_local_dir(str(repo_root), remote_path="/app")
        .run_commands(
            "python -m pip install --upgrade pip setuptools wheel",
            "if [ -f /app/requirements.txt ]; then python -m pip install -r /app/requirements.txt; fi",
            "if [ -f /app/requirements_dev.txt ]; then python -m pip install -r /app/requirements_dev.txt; fi",
        )
    )

    app = modal.App.lookup("hyperagents-meta-transfer", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        *command,
        image=image,
        gpu=args.gpu,
        timeout=args.timeout * 60,
        workdir="/app",
        app=app,
    )

    stdout_thread = threading.Thread(target=_stream_lines, args=(sandbox.stdout, sys.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_stream_lines, args=(sandbox.stderr, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    sandbox.wait(raise_on_termination=False)
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    return sandbox.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
