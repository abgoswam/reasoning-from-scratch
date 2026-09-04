"""Refresh the status of ledger entries by asking Amulet.

RUNS.md records what was submitted; it cannot know what happened afterwards.
This queries `amlt status` for each entry and writes the current state back,
skipping entries that already reached a terminal state.
"""

import argparse
import datetime
import re
import subprocess

TERMINAL = {"pass", "failed", "killed"}
STATES = TERMINAL | {"preparing", "queued", "running", "paused"}

ENTRY = re.compile(r"^## (?P<ts>.+?) — `(?P<job>[^`]+)`$", re.M)
EXPERIMENT = re.compile(r"^- experiment: `(?P<exp>[^`]+)`", re.M)
STATUS_LINE = re.compile(r"^- status: .*$", re.M)


def _query(experiment: str, job: str) -> str:
    out = subprocess.run(
        ["amlt", "status", experiment, f":{job}"],
        capture_output=True, text=True, timeout=180,
    )
    for line in (out.stdout + out.stderr).splitlines():
        if job in line:
            for token in line.split():
                if token in STATES:
                    return token
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="RUNS.md")
    parser.add_argument("--all", action="store_true", help="Re-query terminal entries too")
    args = parser.parse_args()

    with open(args.ledger, encoding="utf-8") as handle:
        text = handle.read()

    starts = [m.start() for m in ENTRY.finditer(text)]
    if not starts:
        print("no entries")
        return

    bounds = list(zip(starts, starts[1:] + [len(text)]))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out, cursor = [], 0

    for start, end in bounds:
        out.append(text[cursor:start])
        block = text[start:end]
        job = ENTRY.search(block).group("job")
        exp_match = EXPERIMENT.search(block)

        existing = STATUS_LINE.search(block)
        current = existing.group(0).split("**")[1].split()[0] if existing and "**" in existing.group(0) else None

        if not exp_match:
            print(f"{job:34} skipped (no experiment recorded)")
        elif current in TERMINAL and not args.all:
            print(f"{job:34} {current} (terminal, skipped)")
        else:
            status = _query(exp_match.group("exp"), job)
            line = f"- status: **{status}** (as of {stamp})"
            block = STATUS_LINE.sub(line, block) if existing else block.replace(
                exp_match.group(0), exp_match.group(0) + "\n" + line, 1
            )
            print(f"{job:34} {current or '-'} -> {status}")

        out.append(block)
        cursor = end

    out.append(text[cursor:])
    with open(args.ledger, "w", encoding="utf-8") as handle:
        handle.write("".join(out))


if __name__ == "__main__":
    main()
