#!/usr/bin/env bash
# phitrain variant of fresh_node_setup.sh, for a box submitted from
# interactive_fast_phitrain.yaml (the TEAM ACR image).
#
# THIS FILE NEVER RUNS ON THE LAPTOP. interactive_fast_phitrain.yaml has
# `code.local_dir: $CONFIG_DIR/src`, so submitting the job uploads this src/ directory and the
# script arrives on the node at /scratch/amlt_code/fresh_node_setup_phitrain.sh. The copy you
# are reading is the source you edit; the node runs the uploaded copy.
#
# AUTOMATED PATH: ../bootstrap_phitrain_node.sh runs steps 2-7 for you from the laptop --
#   ./bootstrap_phitrain_node.sh <node-alias>
# The table below is the manual equivalent, and is what the driver does step by step.
#
#   STEP  WHERE   USER       COMMAND
#   ----  ------  ---------  ---------------------------------------------------------------
#    1    laptop  you        cd .../hello_world_amulet
#                            CONFIG=interactive_fast_phitrain.yaml ./run.sh interactive-fast-phitrain
#                            (the CONFIG= prefix is required -- run.sh defaults to
#                             interactive_fast.yaml, which is a DIFFERENT box)
#    2    laptop  you        gh auth status        # account must be one with microsoft org access
#    3    NODE    root       bash /scratch/amlt_code/fresh_node_setup_phitrain.sh   <-- THIS SCRIPT
#    4    laptop  you        gh auth token | ssh <node-alias> 'cat > /home/aiscuser/.git-token ...'
#    5    NODE    aiscuser   git clone ...          (step b below)
#    6    NODE    aiscuser   pip install -e ...     (step c below)
#    7    NODE    aiscuser   source <venv>/bin/activate   (step d below)
#
# ALREADY HAVE A NODE? Steps 1-2 are what GOT you the node, so they are done. From a fresh
# terminal on a running box the remaining work is, in order:
#
#     step 3  on the node, as root      <- run this script
#     step 4  back on the laptop        <- send the token over
#     steps 5-7  on the node, as aiscuser
#
# That is it. Start at step 3 and follow what the script prints.
#
# Step 3 is the only step that is a script, and it is the only step that runs as root -- it
# writes machine-level VS Code settings and chowns /home/aiscuser. It is idempotent; re-running
# is safe. When it finishes it PRINTS steps 4-7 as copy-pasteable text, so the node tells you
# what comes next and you do not need this table in front of you.
#
# Steps 4-7 stay manual because they need your GitHub token, which is not something a script
# should be handling. Everything after step 3 runs as aiscuser: a clone done as root lands
# root-owned and the editable install then fails on permissions.
#
# THE TEAM IMAGE HAS NO CONDA. Verified on the node 2026-09-14:
#
#   conda env list     -> empty            which python -> /usr/bin/python (3.12.3)
#   torch 2.10.0+cu130   vllm 0.18.1.dev0+cu130   torchao 0.17.0
#   flash_attn 2.8.4     deepspeed 0.18.9         transformers 5.5.4
#
# Everything phitrain needs is in SYSTEM site-packages, not an env. So this variant does not
# create a conda env (SKIP_ENV=1 on the shared script) and builds a
# `venv --system-site-packages` instead: aiscuser-owned, editable installs land inside it, and
# torch/vllm/flash-attn are inherited rather than reinstalled. Rebuilding those with pip against
# the image's CUDA is not a fight worth having.
#
# Alternative, for reference: phitrain_baltic.yaml :smoke uses PYTHONUSERBASE=/tmp/amlt-user.
# That works, but it is only visible to processes with that variable set -- fresh terminals and
# notebook kernels do not have it, which is the same class of trap as `pip install --user`.
#
# REPO_URL="" -- the shared script cannot clone microsoft/aifsdk: it is PRIVATE and the node has
# no credential helper, no gh and no ~/.netrc. The clone is done by hand, as aiscuser, using a
# token carried over from the laptop. Recipe is printed at the end and was walked end to end on
# 2026-09-14; the traps found doing it are recorded there.
set -uo pipefail

# --- guard: this must run ON THE NODE, as root -----------------------------------------------
# Without this, running the script on the laptop would have fresh_node_setup.sh chown
# /home/aiscuser and write machine-level VS Code settings on your own machine.
if [ ! -d /scratch/amlt_code ] || ! id aiscuser >/dev/null 2>&1; then
  echo "REFUSING: this does not look like an Amulet node." >&2
  echo "  /scratch/amlt_code exists : $([ -d /scratch/amlt_code ] && echo yes || echo NO)" >&2
  echo "  aiscuser exists           : $(id aiscuser >/dev/null 2>&1 && echo yes || echo NO)" >&2
  echo "Run it on the node instead:  bash /scratch/amlt_code/$(basename "$0")" >&2
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "REFUSING: run as root (you are $(whoami)). Steps 5-7 are the aiscuser ones." >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/scratch/venvs/phitrain}"
SYSPY="$(command -v python3)"
CODE=/scratch/amlt_code

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
skip() { printf '  --    %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }

# steps 1-5 (keepalive, dotfiles, VS Code settings, mounts) are identical; skip 6-10
# DEFAULT_PY: this track has no conda env, so point VS Code at the venv step 11 builds.
SKIP_ENV=1 REPO_URL="" QUIET_MANUAL=1 DEFAULT_PY="$VENV/bin/python" \
  bash "$HERE/fresh_node_setup.sh" "$@"

echo "== 11. venv with system site-packages =="
if [ -x "$VENV/bin/python" ]; then
  skip "$VENV already exists"
else
  mkdir -p "$(dirname "$VENV")"
  "$SYSPY" -m venv --system-site-packages "$VENV" \
    && ok "created $VENV from $SYSPY ($("$SYSPY" -V 2>&1))" || { warn "venv creation failed"; exit 1; }
fi
if [ "$(stat -c %U "$VENV")" = "aiscuser" ]; then
  skip "already owned by aiscuser"
else
  chown -R aiscuser:aiscuser "$VENV" && ok "chowned to aiscuser"
fi
su aiscuser -c "$VENV/bin/python -m pip install --quiet ipykernel && \
  $VENV/bin/python -m ipykernel install --user --name phitrain --display-name 'Python (phitrain)'" \
  > /dev/null 2>&1 && ok "ipykernel + kernel 'Python (phitrain)' registered" || warn "kernel registration failed"

# git's safe.directory check fires on the clone below because the repo is created under
# /scratch by aiscuser while some commands get run from a root shell. Pre-authorise it for
# both users so `git status`/`sparse-checkout list` do not die with "dubious ownership".
for u in root aiscuser; do
  su "$u" -c "git config --global --add safe.directory $CODE/aifsdk" 2>/dev/null
done
ok "git safe.directory set for $CODE/aifsdk (root + aiscuser)"

cat <<EOF

== phitrain: clone and install ==  (steps 4-7; step 3 is done)

  Step a runs on your LAPTOP. Steps b-e run HERE, on the node, as aiscuser -- so before b:

      su - aiscuser          # then check: whoami   ->   aiscuser

  A clone done as root lands root-owned and the editable install then fails on permissions.

  --- a. [LAPTOP] send a GitHub token over to the node ------------------------------------
  aifsdk is private and the node has no gh, so reuse the laptop's gh token. In WSL:

    gh auth status                 # the account MUST have microsoft org access
    gh auth token | ssh <node-alias> 'cat > /home/aiscuser/.git-token \\
      && chown aiscuser:aiscuser /home/aiscuser/.git-token && chmod 600 /home/aiscuser/.git-token'

  TRAP (cost ~30 min on 2026-09-14): a token from an account without microsoft org access
  clones with \`remote: Repository not found\` / \`fatal: repository ... not found\`. That is a
  404 standing in for a 403 -- GitHub hides private repos you cannot see. It is NOT a bad
  token and NOT a typo in the URL. Check with \`gh api repos/microsoft/aifsdk --jq .full_name\`
  in WSL first; if it 404s there too, switch gh accounts (\`gh auth switch\`) and re-send the
  token, since the one already on the node is now stale.

  Never cat or paste the token itself.

  --- b. [NODE] sparse clone ------------------------------------------------------------------------
    cd $CODE
    TOKEN=\$(cat /home/aiscuser/.git-token)
    git clone --filter=blob:none --sparse https://x-access-token:\$TOKEN@github.com/microsoft/aifsdk.git
    cd aifsdk
    git sparse-checkout set phitrain phiagent
    git remote set-url origin https://github.com/microsoft/aifsdk.git   # drop token from .git/config
    printf 'https://x-access-token:%s@github.com\\n' "\$(cat ~/.git-token)" > ~/.git-credentials
    chmod 600 ~/.git-credentials
    git config --global credential.helper store     # 'store' does nothing without this line

  The last three lines matter: this is a --filter=blob:none partial clone, so any later
  checkout or fetch has to pull blobs from origin. Without a credential helper the next
  'git checkout <branch>' stops at an interactive "Username for 'https://github.com':" prompt.

  --filter=blob:none --sparse matters: aifsdk is a ~27 GB working tree and you need two
  directories. Expect it to sit quiet for a while fetching history metadata.
  The set-url line strips the token back out of .git/config; the helper serves it from then on,
  for fetch, checkout and push alike.

  --- c. [NODE] editable install --------------------------------------------------------------------
    cd $CODE/aifsdk                                      # <- MUST be the repo root
    $VENV/bin/python -m pip install -e phiagent -e phitrain --no-deps
    $VENV/bin/python -m pip install \\
      'apache-tvm-ffi<0.1.10' 'kernels==0.14.1' 'lightning==2.6.5' \\
      'omegaconf==2.4.0.dev14' 'transformers==5.9.0' 'wandb==0.26.0' \\
      sentencepiece tiktoken

  The paths are relative, so running this from $CODE silently does the wrong thing.
  --no-deps on the editable installs is load-bearing: without it pip resolves phitrain's full
  tree and will replace the inherited torch and vllm.
  Pins copied from phitrain_baltic.yaml :smoke -- the combination proven against this image.

  How to tell the second command did not run: \`pip list\` shows lightning 2.6.1 and
  omegaconf 2.0.0. Those are the image's SYSTEM versions showing through --system-site-packages,
  not our pins, and they look plausible enough to miss.

  --- d. [NODE] using phitrain-cli ------------------------------------------------------------------
    source $VENV/bin/activate      # then: phitrain-cli --help

  The console script is only ever at $VENV/bin/phitrain-cli, and that directory is not on PATH
  unless the venv is activated -- calling the venv python by full path does not put it there.
  Without activating, use the full path or the module form:

    $VENV/bin/python -m phitrain.cli.interface --help

  The module form is what omni-eval itself switched to in PR #3656 (runner.py now calls
  sys.executable -m phitrain.cli.interface) for exactly this reason: the console script's
  location is not portable.

  --- e. [NODE] verify ------------------------------------------------------------------------------
    $VENV/bin/python -c 'import torch, torchao, transformers, vllm, wandb; print(torch.__version__, vllm.__version__)'
    $VENV/bin/python -c 'import torch; print("CUDA", torch.cuda.is_available(), torch.cuda.device_count())'
    $VENV/bin/python -c 'import phitrain, phiagent; print(phitrain.__file__)'   # must be under $CODE
    cd phitrain && $VENV/bin/python -c 'import recipes.rl.tool_agent.train; print("recipe import OK")'

  VS Code interpreter: preselected to $VENV/bin/python by step 5 (reload the window if it
  still asks). Notebook kernel: 'Python (phitrain)'.
EOF
