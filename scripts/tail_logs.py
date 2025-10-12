#!/usr/bin/env python3
"""Tail docker compose logs with colored, emoji-enhanced output."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Iterable, List, Optional, Tuple


LOG_PATTERN = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b", re.IGNORECASE)

EMOJIS = {
    "DEBUG": "🐞",
    "INFO": "ℹ️ ",
    "WARNING": "⚠️ ",
    "ERROR": "❌",
    "CRITICAL": "🚨",
}

COLORS = {
    "DEBUG": "\033[90m",  # Bright black (gray)
    "INFO": "\033[94m",  # Bright blue
    "WARNING": "\033[93m",  # Bright yellow
    "ERROR": "\033[91m",  # Bright red
    "CRITICAL": "\033[41m\033[97m",  # White on red background
}

RESET = "\033[0m"


def _detect_compose_command() -> Optional[List[str]]:
    """Return the preferred docker compose command available on this system."""
    if shutil.which("docker"):
        # Prefer the integrated `docker compose` subcommand if available.
        try:
            subprocess.run(
                ["docker", "compose", "version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        else:
            return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


def _build_command(base_cmd: List[str], services: Iterable[str]) -> List[str]:
    """Build the docker compose logs command."""
    cmd = base_cmd + ["logs", "--follow", "--tail", "100"]
    cmd.extend(services)
    return cmd


def _colorize(line: str, use_color: bool) -> str:
    """Apply emojis and ANSI colors based on detected log level."""
    match = LOG_PATTERN.search(line)
    if not match:
        return line
    level = match.group(1).upper()
    emoji = EMOJIS.get(level, "")
    color = COLORS.get(level, "")
    if not use_color:
        return f"{emoji} {line}" if emoji else line
    colored = f"{color}{emoji} {line}{RESET}" if emoji else f"{color}{line}{RESET}"
    return colored


def tail_logs(services: Iterable[str]) -> int:
    """Stream docker compose logs with decorated output."""
    base_cmd = _detect_compose_command()
    if base_cmd is None:
        print("Unable to find `docker compose` or `docker-compose` on PATH.", file=sys.stderr)
        return 1

    cmd = _build_command(base_cmd, services)
    pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
    print(f"Streaming logs via: {pretty_cmd}\n", file=sys.stderr)

    use_color = sys.stdout.isatty()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def _handle_interrupt(signum, frame):
        if process.poll() is None:
            process.terminate()

    original_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            print(_colorize(line, use_color))
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    return process.returncode or 0


def parse_args(argv: Optional[Iterable[str]] = None) -> Tuple[List[str]]:
    parser = argparse.ArgumentParser(
        description="Tail docker compose logs with emoji and color-coded levels.",
    )
    parser.add_argument(
        "services",
        nargs="*",
        help="Optional service names to filter (defaults to all services).",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    return parsed.services


def main(argv: Optional[Iterable[str]] = None) -> int:
    services = parse_args(argv)
    return tail_logs(services)


if __name__ == "__main__":
    raise SystemExit(main())
