"""Emit a ~/.ssh/config Host block for a running Amulet job, for VS Code Remote-SSH.

Two targets:
  (default)   WSL's ~/.ssh/config -- for VS Code running inside WSL, or plain ssh.
  --windows   C:\\Users\\<user>\\.ssh\\config -- for VS Code running on Windows.

The Windows variant rewrites three fields, because Amulet's block is WSL-bound:
the ProxyCommand runs /opt/az/bin/python3 against the `az ml` extension under
~/.azure/, neither of which exists on Windows. Prefixing `wsl.exe -e` delegates
just the tunnel to WSL (and reuses WSL's `az login`), while ssh itself, the key,
and VS Code stay on Windows.

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


def _win_user(explicit: str | None) -> str:
    """Windows username -- it differs from the WSL one on this box."""
    if explicit:
        return explicit
    try:
        out = subprocess.run(["cmd.exe", "/c", "echo %USERNAME%"], capture_output=True,
                             text=True, timeout=15).stdout.strip()
        if out and "%" not in out:
            return out
    except Exception:
        pass
    raise SystemExit("Could not detect the Windows username -- pass --win-user")


def _to_windows(parsed: dict[str, str], win_user: str) -> dict[str, str]:
    """Rewrite a WSL-shaped block for Windows ssh."""
    out = dict(parsed)
    # The connector is a WSL path; run it through WSL rather than porting it.
    out["proxy"] = f"wsl.exe -e {parsed['proxy']}"
    # Same key, Windows-side copy. Verified identical by fingerprint.
    if "identity" in parsed:
        out["identity"] = f"C:\\Users\\{win_user}\\.ssh\\{Path(parsed['identity']).name}"
    return out


def _block(alias: str, parsed: dict[str, str], windows: bool = False) -> str:
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
        f"    UserKnownHostsFile {'NUL' if windows else '/dev/null'}",
        "    ServerAliveInterval 30",
        END.format(alias=alias),
    ]
    return "\n".join(lines)


def _write(alias: str, block: str, path: Path | None = None) -> Path:
    path = path or Path.home() / ".ssh" / "config"
    path.parent.mkdir(mode=0o700, exist_ok=True)
    existing = path.read_text() if path.exists() else ""

    pattern = re.compile(
        re.escape(BEGIN.format(alias=alias)) + r".*?" + re.escape(END.format(alias=alias)) + r"\n?",
        re.DOTALL,
    )
    updated = pattern.sub("", existing).rstrip()
    updated = f"{updated}\n\n{block}\n" if updated else f"{block}\n"

    path.write_text(updated)
    try:
        path.chmod(0o600)
    except (PermissionError, OSError):
        pass  # DrvFs (/mnt/c) does not hold Unix modes
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("job")
    parser.add_argument("--alias", default=None, help="Host alias; defaults to amlt-<job>")
    parser.add_argument("--write", action="store_true", help="Update ~/.ssh/config in place")
    parser.add_argument("--dump", default=None, help="Write Amulet's raw output to this file")
    parser.add_argument("--windows", action="store_true",
                        help="Emit a block for Windows ssh / VS Code on Windows (wraps the "
                             "ProxyCommand in wsl.exe and uses the Windows-side key).")
    parser.add_argument("--win-user", default=None,
                        help="Windows username; auto-detected via cmd.exe when omitted.")
    args = parser.parse_args()

    alias = args.alias or f"amlt-{args.job}"
    parsed = _parse(_capture_ssh_command(args.experiment, args.job, args.dump))

    if args.windows:
        win_user = _win_user(args.win_user)
        parsed = _to_windows(parsed, win_user)
        target = Path(f"/mnt/c/Users/{win_user}/.ssh/config")
    else:
        target = None

    block = _block(alias, parsed, windows=args.windows)

    if args.write:
        path = _write(alias, block, target)
        print(f"\nUpdated {path}\n")
        print(block)
        print(f"\nVS Code: Remote-SSH: Connect to Host... -> {alias}")
    else:
        where = target or (Path.home() / ".ssh" / "config")
        print(f"\n{block}\n")
        print(f"Add that to {where}, or re-run with --write. Then in VS Code:")
        print(f"  Remote-SSH: Connect to Host... -> {alias}")


if __name__ == "__main__":
    main()
