#!/usr/bin/env bash
# Submit one job from train.yaml into the shared ch07 experiment, and record it
# in RUNS.md (newest first).
#
#   ./run.sh smoke
#   ./run.sh tracking -d "500 steps, run 1"
#   EXPERIMENT=agoswami-rfs-scratch ./run.sh smoke
#
# Job names must be unique within an experiment, so each run gets a timestamp
# suffix via Amulet's :job=alias syntax.

set -euo pipefail
cd "$(dirname "$0")"

JOB="${1:?usage: ./run.sh <smoke|smoke-wandb|baseline|tracking> [extra amlt run args]}"
shift

EXPERIMENT="${EXPERIMENT:-agoswami-rfs-ch07}"
JOB_NAME="${JOB}-$(date +%Y%m%d-%H%M%S)"
LEDGER="RUNS.md"

# Amulet resolves every single-$ variable in train.yaml at config-load time,
# across ALL jobs whichever one is selected -- so W&B vars are needed even for
# :smoke, which passes --no_wandb.
set -a; . ./.env; set +a

# Amulet refuses to submit unless the image has a current vulnerability scan.
export FEDRAMP_SCANNER_SCAN_MODE="${FEDRAMP_SCANNER_SCAN_MODE:-none}"

CMD=(amlt run train.yaml ":${JOB}=${JOB_NAME}" "$EXPERIMENT" --sla Premium --yes "$@")
echo "+ ${CMD[*]}"

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
    f'\n## {os.environ["L_TS"]} \u2014 `{os.environ["L_JOB_NAME"]}`\n',
    f'- experiment: `{os.environ["L_EXPERIMENT"]}`\n',
    f'- status: **submitted** (as of {os.environ["L_TS"]})\n',
    f'- config job: `{os.environ["L_JOB"]}`\n',
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
