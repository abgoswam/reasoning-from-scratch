"""Emit a ~/.ssh/config Host block for a running Amulet job, for VS Code Remote-SSH.

`amlt ssh` builds an ordinary ssh command whose websocket tunnel is just a
ProxyCommand. This asks Amulet for that command, parses it, and reshapes it into
a Host block that VS Code Remote-SSH can consume.

The proxy endpoint is minted per submission, so the block is only valid for the
job it was generated from. Re-run this after every resubmit.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

BEGIN = "# >>> amlt {alias} >>>"
END = "# <<< amlt {alias} <<<"


def _capture_ssh_command(experiment: str, job: str, dump: str | None = None) -> list[str]:
    cmd = ["amlt", "-vvv", "ssh", experiment, f":{job}", "-c", "true"]
    print(f"$ {shlex.join(cmd)}", file=sys.stderr)

    # Amulet renders logs through rich, which hard-wraps to the terminal width and
    # would split the quoted ProxyCommand across lines.
    env = {**os.environ, "COLUMNS": "100000", "TERM": "dumb", "NO_COLOR": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    blob = ANSI.sub("", proc.stdout + proc.stderr)

    if dump:
        Path(dump).write_text(blob)
        print(f"raw output written to {dump}", file=sys.stderr)

    match = re.search(r"ssh command:\s*(.+)", blob)
    if not match:
        print(blob.strip()[-2000:], file=sys.stderr)
        raise SystemExit(
            "\nCould not find the ssh command in Amulet's output. The job must be "
            "running AND have been submitted with `amlt run -i`."
        )

    line = match.group(1).strip()
    try:
        return shlex.split(line)
    except ValueError as exc:
        print(f"\nCaptured line:\n{line}\n", file=sys.stderr)
        raise SystemExit(
            f"Could not parse that as a shell command ({exc}). It looks truncated — "
            "re-run with --dump raw.txt and inspect the 'ssh command:' line."
        )


def _parse(argv: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for i, token in enumerate(argv):
        if token == "-o" and i + 1 < len(argv) and argv[i + 1].startswith("ProxyCommand="):
            parsed["proxy"] = argv[i + 1].split("=", 1)[1]
        elif token == "-i" and i + 1 < len(argv):
            parsed["identity"] = argv[i + 1]
        elif "@" in token and not token.startswith("-"):
            user, _, host = token.partition("@")
            parsed["user"] = user
            parsed["host"] = host

    missing = {"proxy", "host", "user"} - parsed.keys()
    if missing:
        raise SystemExit(f"Could not parse {', '.join(sorted(missing))} from: {shlex.join(argv)}")
    return parsed


def _block(alias: str, parsed: dict[str, str]) -> str:
    lines = [
        BEGIN.format(alias=alias),
        f"Host {alias}",
        f"    HostName {parsed['host']}",
        f"    User {parsed['user']}",
    ]
    if "identity" in parsed:
        lines.append(f"    IdentityFile {parsed['identity']}")
    lines += [
        f"    ProxyCommand {parsed['proxy']}",
        "    StrictHostKeyChecking no",
        "    UserKnownHostsFile /dev/null",
        "    ServerAliveInterval 30",
        END.format(alias=alias),
    ]
    return "\n".join(lines)


def _write(alias: str, block: str) -> Path:
    path = Path.home() / ".ssh" / "config"
    path.parent.mkdir(mode=0o700, exist_ok=True)
    existing = path.read_text() if path.exists() else ""

    pattern = re.compile(
        re.escape(BEGIN.format(alias=alias)) + r".*?" + re.escape(END.format(alias=alias)) + r"\n?",
        re.DOTALL,
    )
    updated = pattern.sub("", existing).rstrip()
    updated = f"{updated}\n\n{block}\n" if updated else f"{block}\n"

    path.write_text(updated)
    path.chmod(0o600)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("job")
    parser.add_argument("--alias", default=None, help="Host alias; defaults to amlt-<job>")
    parser.add_argument("--write", action="store_true", help="Update ~/.ssh/config in place")
    parser.add_argument("--dump", default=None, help="Write Amulet's raw output to this file")
    args = parser.parse_args()

    alias = args.alias or f"amlt-{args.job}"
    block = _block(alias, _parse(_capture_ssh_command(args.experiment, args.job, args.dump)))

    if args.write:
        path = _write(alias, block)
        print(f"\nUpdated {path}\n")
        print(block)
        print(f"\nVS Code: Remote-SSH: Connect to Host... -> {alias}")
    else:
        print(f"\n{block}\n")
        print(f"Add that to ~/.ssh/config, or re-run with --write. Then in VS Code:")
        print(f"  Remote-SSH: Connect to Host... -> {alias}")


if __name__ == "__main__":
    main()
