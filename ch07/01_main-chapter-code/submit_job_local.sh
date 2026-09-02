#!/usr/bin/env bash
# Submit one GRPO training job.
#
# Uncomment exactly one CMD line below, then:  ./submit_job_local.sh
# Leave the old ones commented out -- together with jobs.log they are the
# record of what has been run.
#
# Check which GPUs are free first (nvidia-smi); a job already training on
# cuda:1 will OOM a second job pointed at the same device.

set -euo pipefail
cd "$(dirname "$0")"

# 7.2 baseline -- num_rollouts 4, because 8 x 1024 tokens OOM'd this 44 GB card
# at step 75. The cluster run uses 8 to match the book.
#CMD="python submit_02_baseline.py --steps 500 --max_new_tokens 1024 --num_rollouts 4 --device cuda:1"

# 7.3 + advantage stats and entropy -- adds adv_avg, adv_std, entropy_avg
CMD="python submit_03_tracking.py --steps 500 --max_new_tokens 1024 --num_rollouts 4 --device cuda:2"

printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CMD" >> jobs.log
echo "+ $CMD"
exec $CMD
