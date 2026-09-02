#!/usr/bin/env bash
# Same as submit_job_local.sh, but runs CMD on the bonete cluster with 1 GPU.

set -euo pipefail
cd "$(dirname "$0")"

CLUSTER_SUBMIT="${CLUSTER_SUBMIT:-/home/agoswami/_hackerreborn/aifsdk/clusters/lambda/submission/submit_job.sh}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

export PRIORITY="${PRIORITY:-p0}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"
export PROJECT_NAME="${PROJECT_NAME:-aion}"
export USER_ALIAS="${USER_ALIAS:-agoswami}"

# .env is gitignored, so it is not uploaded; feed it to the submitter instead,
# which injects WANDB_API_KEY/WANDB_BASE_URL into the pod.
set -a; . ./.env; set +a
export WANDB_HOST="$WANDB_BASE_URL"

#CMD="python submit_02_baseline.py --steps 500 --max_new_tokens 1024 --num_rollouts 8 --eval_on_checkpoint 128"
CMD="python submit_03_tracking.py --steps 500 --max_new_tokens 1024 --num_rollouts 8 --eval_on_checkpoint 128"

# Job name carries the script, e.g. agoswami-p0-aion-submit-02-baseline-<suffix>.
TAG="$(awk '{print $2}' <<<"$CMD")"; TAG="${TAG%.py}"; TAG="${TAG//_/-}"
export JOB_NAME="${USER_ALIAS}-${PRIORITY}-${PROJECT_NAME}-${TAG}"

read -r -d '' JOB_CMD <<EOF || true
set -euo pipefail
pip install -e . --no-deps --quiet
pip install --quiet tokenizers requests wandb sympy

cd ch07/01_main-chapter-code
$CMD --out_dir "\$OUTPUT_DIR"
EOF

printf '%s  [cluster] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CMD" >> jobs.log
echo "+ [cluster] $CMD"

exec bash "$CLUSTER_SUBMIT" \
    --upload "$REPO_ROOT" \
    --node 1 --gpu-per-node 1 --cpu 8 --memory 64Gi --shm 8Gi \
    --cmd "$JOB_CMD"
