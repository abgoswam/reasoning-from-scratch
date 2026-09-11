#!/usr/bin/env bash
# Submit one job from interactive_fast.yaml into the shared interactive
# experiment, and record it in RUNS.md (newest first).
#
#   ./run.sh interactive-fast
#   CONFIG=interactive.yaml ./run.sh interactive          # team-image variant
#   EXPERIMENT=agoswami-scratch ./run.sh interactive-fast
#   DRY_RUN=1 ./run.sh interactive-fast                   # print the command, submit nothing
#
# Project: agoswami-hello (aifrontierssadata/data). Bind this directory with:
#   amlt project checkout agoswami-hello aifrontierssadata data
#
# Job names must be unique within an experiment, so each run gets a timestamp
# suffix via Amulet's :job=alias syntax.

set -euo pipefail
cd "$(dirname "$0")"

JOB="${1:?usage: ./run.sh <interactive-fast|interactive> [extra amlt run args]}"
shift

PROJECT="agoswami-hello"
EXPERIMENT="${EXPERIMENT:-agoswami-interactive}"
CONFIG="${CONFIG:-interactive_fast.yaml}"
# Prefixed with the alias so the job is identifiable in shared views -- notably
# W&B: phitrain passes no name= to wandb.init (handlers.py:169), so the SDK falls
# back to WANDB_NAME, which Amulet sets to this job name. Runs land in the team
# project, so the prefix is how you find yours.
USER_ALIAS="${USER_ALIAS:-agoswami}"
JOB_NAME="${USER_ALIAS}-${JOB}-$(date +%Y%m%d-%H%M%S)"
LEDGER="RUNS.md"

# Amulet refuses to submit unless the image has a current vulnerability scan.
export FEDRAMP_SCANNER_SCAN_MODE="${FEDRAMP_SCANNER_SCAN_MODE:-none}"

# -i is REQUIRED for `amlt ssh` to work against this job.
CMD=(amlt run "$CONFIG" ":${JOB}=${JOB_NAME}" "$EXPERIMENT" -i --sla Premium --yes "$@")
echo "+ ${CMD[*]}"

# DRY_RUN=1 ./run.sh <job> -- print the assembled command and stop.
if [ -n "${DRY_RUN:-}" ]; then
  echo "(DRY_RUN set -- not submitting)"
  exit 0
fi

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
set +e
"${CMD[@]}" 2>&1 | tee "$OUT"
RC=${PIPESTATUS[0]}
set -e

# A failed submit creates no job and no artifacts, so there is nothing to track.
if [ "$RC" -ne 0 ]; then
  echo "submit failed (rc=$RC) -- nothing written to $LEDGER"
  exit "$RC"
fi

{
flock 9

L_TS="$(date '+%Y-%m-%d %H:%M:%S %Z')" \
L_JOB="$JOB" \
L_JOB_NAME="$JOB_NAME" \
L_PROJECT="$PROJECT" \
L_CONFIG="$CONFIG" \
L_EXPERIMENT="$EXPERIMENT" \
L_PORTAL="$(grep -oE 'https://aka\.ms/amlt\?q=[A-Za-z0-9]+' "$OUT" | head -1 || true)" \
L_GIT_SHA="$(git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)" \
L_GIT_STATE="$([ -z "$(git -C .. status --porcelain 2>/dev/null)" ] && echo clean || echo dirty)" \
L_CMD="$(printf '%q ' "${CMD[@]}")" \
L_FILE="$LEDGER" \
python3 <<'PY'
import os, pathlib

HEADER = "# Run ledger\n\nNewest first. Written by run.sh.\n"
portal = os.environ["L_PORTAL"]

lines = [
    f'\n## {os.environ["L_TS"]} — `{os.environ["L_JOB_NAME"]}`\n',
    f'- project: `{os.environ["L_PROJECT"]}` (aifrontierssadata/data)\n',
    f'- experiment: `{os.environ["L_EXPERIMENT"]}`\n',
    f'- config: `{os.environ["L_CONFIG"]}` job `{os.environ["L_JOB"]}`\n',
    f'- status: **submitted** (as of {os.environ["L_TS"]})\n',
]
if portal:
    lines.append(f"- portal: {portal}\n")
lines += [
    f'- code: `{os.environ["L_GIT_SHA"]}` ({os.environ["L_GIT_STATE"]})\n',
    f'- monitor: `amlt status {os.environ["L_EXPERIMENT"]} :{os.environ["L_JOB_NAME"]}`\n',
    f'\n```bash\n{os.environ["L_CMD"]}\n```\n',
]
entry = "".join(lines)

path = pathlib.Path(os.environ["L_FILE"])
body = path.read_text(encoding="utf-8")[len(HEADER):] if path.exists() else ""
path.write_text(HEADER + entry + body, encoding="utf-8")
PY

} 9>"$LEDGER.lock"

echo
echo "ledger: $(pwd)/$LEDGER"
exit "$RC"
