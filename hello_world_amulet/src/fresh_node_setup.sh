#!/usr/bin/env bash
# Set up a fresh (or just-restarted) interactive box. Idempotent: safe to re-run.
#
#   bash /scratch/amlt_code/fresh_node_setup.sh                 # run it as ROOT
#   ENV_NAME=ptca4 bash /scratch/amlt_code/fresh_node_setup.sh   # build a differently-named env
#   bash /scratch/amlt_code/fresh_node_setup_phitrain.sh         # phitrain variant (clones ptca)
#
# A VS Code terminal and an ssh session both land as root; the script re-invokes itself as
# aiscuser for steps 6-7 so nothing root-owned is left in the env or the repo.
#
# Lives in hello_world_amulet/src/ = the job's code.local_dir, so it is uploaded with every
# box and is already on the node at /scratch/amlt_code/.
#
# WHAT IT DOES
#
#   #  step                     runs as    writes to                               skipped when
#   -  -----------------------  ---------  --------------------------------------  --------------------
#   1  GPU keepalive            caller     /scratch/amlt_code/keepalive.log        GPU already >100 MiB
#   2  job env for shells       caller     $HOME/.bashrc                           line already present
#   3  login shells read bashrc caller     /home/aiscuser/.profile                 line already present
#   4  report mounts            caller     nothing (read-only)                     never
#   5  VS Code machine settings root       ~/.vscode-server/data/Machine           file already present
#   6  create the env           root       /opt/conda/envs/$ENV_NAME (clean 3.10)  env already exists
#   7  clone the repo           aiscuser   /scratch/amlt_code/reasoning-from-...   .git already exists
#   8  pip install into the env aiscuser   /opt/conda/envs/$ENV_NAME               imports already work
#   9  register jupyter kernel  aiscuser   ~/.local/share/jupyter/kernels          kernel already listed
#  10  install claude           aiscuser   ~/.local/bin/claude                     binary already there
#
#   MANUAL, printed at the end:  W&B key in .env  |  `claude` login (install is step 10)
#
# WHY A SEPARATE, CLEAN ptca3 ENV
#
# `ptca` is the image's env: root-owned and shared. Installing into it needs root, and root then
# drops root-owned egg-info into your repo and root-owned files into the env that aiscuser
# cannot later modify. `--user` is no better: a user-site install is only visible when $HOME
# happens to be /home/aiscuser, so it works in a VS Code terminal and fails over ssh with
# ModuleNotFoundError (observed 2026-09-14).
#
# ptca3 is created CLEAN (`conda create -n ptca3 python=3.10`), not cloned, then chowned to
# aiscuser. Clean rather than a clone so the env contains exactly what this repo declares and
# nothing inherited: no surprise pins, no half-configured extras, and a dependency problem is
# reproducible from the pyproject alone. The cost is that pip pulls torch and its CUDA libs
# from PyPI (~3 GB, a few minutes) instead of reusing the image's build -- so the FIRST run of
# step 7 is slow, and `ptca` is left exactly as the image shipped it.
#
# python=3.10 matches the image, and the repo allows >=3.10,<3.15.
#
# PYTHONNOUSERSITE=1 is set for every pip call: if a user-site copy of a dependency exists, pip
# treats it as satisfied and silently omits it from the env, which is how the env ended up with
# wandb but not click, then not opentelemetry, then not regex.
#
# Nothing here survives a container restart -- not /scratch, not the env, not the dotfiles.
# Re-run the whole script after one; every step is idempotent.
#
# Full background: hello_world_amulet/fresh-node-setup.md
set -uo pipefail

CONDA=/opt/conda/bin/conda
ENV_NAME="${ENV_NAME:-ptca3}"           # override to validate: ENV_NAME=ptca4 bash ...
# ENV_BASE empty  -> create a CLEAN python=3.10 env (right for reasoning-from-scratch).
# ENV_BASE=ptca   -> CLONE the image env instead. Needed when the work requires vllm / torchao /
#                    flash-attn, which pip cannot reasonably rebuild against the image's CUDA.
#                    Hardlinked within /opt/conda, so ~free despite ptca being 8.4 GB.
ENV_BASE="${ENV_BASE:-}"
ENV=/opt/conda/envs/$ENV_NAME           # ours: clean, aiscuser-owned
PY=$ENV/bin/python
# Only for the keepalive, before our env exists. NOT hardcoded to conda: the team ACR image
# (interactive_fast_phitrain.yaml) has no conda at all -- python 3.12 in system site-packages.
BOOTPY="$( [ -x /opt/conda/envs/ptca/bin/python ] && echo /opt/conda/envs/ptca/bin/python || command -v python3 )"
CODE=/scratch/amlt_code
# REPO_URL empty -> skip the clone and the editable install entirely (steps 7-8), for cases
# where the repo is private and you will clone it by hand.
REPO_URL="${REPO_URL-https://github.com/abgoswam/reasoning-from-scratch.git}"
REPO=$CODE/reasoning-from-scratch
# The repo's own deps come from `pip install -e .` (torch, jupyterlab, tokenizers, nbformat,
# sympy, matplotlib). These are the extras it does not declare.
PKGS="wandb transformers requests ipykernel"

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
skip() { printf '  --    %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }

if [ "${1:-}" = "--as-aiscuser" ]; then STAGE2=1; else STAGE2=""; fi

if [ -z "$STAGE2" ]; then

echo "== 1. keepalive =="
# Check the GPU, never pgrep: `pgrep -f keep_wake` matches this script's own command line,
# and `pkill -f` on it would kill this shell.
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${MEM:-0}" -gt 100 ]; then
  skip "already holding ${MEM} MiB on GPU 0"
else
  cd "$CODE"
  nohup bash -c "while true; do $BOOTPY keep_wake.py; sleep 5; done" \
    > "$CODE/keepalive.log" 2>&1 < /dev/null & disown
  sleep 25
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${MEM:-0}" -gt 100 ] && ok "started, ${MEM} MiB held" || warn "started but GPU idle -- see $CODE/keepalive.log"
fi

echo "== 2. job env for interactive shells =="
# PWD/OLDPWD/SHLVL are filtered: sourcing them assigns $PWD without cd-ing, so the prompt lies
# about the cwd. /tmp/amlt-env holds real credentials -- never cat or commit it.
if grep -q amlt-env "$HOME/.bashrc" 2>/dev/null; then
  skip "$HOME/.bashrc already sources /tmp/amlt-env"
else
  {
    echo 'export PATH="$HOME/.local/bin:$PATH"'
    echo '[ -e /tmp/amlt-env ] && . <(grep -Ev "^declare -x (PWD|OLDPWD|SHLVL)=" /tmp/amlt-env)'
    echo 'export PATH="$HOME/.local/bin:$PATH"'
  } >> "$HOME/.bashrc"
  ok "appended to $HOME/.bashrc"
fi

echo "== 3. login shells read .bashrc =="
PROFILE=/home/aiscuser/.profile
if grep -q 'bashrc' "$PROFILE" 2>/dev/null; then
  skip "$PROFILE already sources .bashrc"
else
  printf '\n# source .bashrc so login shells (su -, ssh) match VS Code terminals\nif [ -n "$BASH" ] && [ -f "$HOME/.bashrc" ]; then\n    . "$HOME/.bashrc"\nfi\n' >> "$PROFILE"
  ok "appended to $PROFILE"
fi

echo "== 3b. aiscuser owns its own dotdirs =="
# On the team ACR image /home/aiscuser/.local is created ROOT-owned before we ever run, so
# `pip install --user`, `ipykernel install --user` and the claude installer all fail with EACCES
# or "Permission denied: /home/aiscuser/.local/share/...". Observed 2026-09-14. Cheap to fix and
# harmless where it is already correct.
FIXED=""
for d in /home/aiscuser/.local /home/aiscuser/.cache /home/aiscuser/.config; do
  mkdir -p "$d" 2>/dev/null
  [ "$(stat -c %U "$d" 2>/dev/null)" = "aiscuser" ] || { chown -R aiscuser:aiscuser "$d" && FIXED="$FIXED $(basename $d)"; }
done
[ -n "$FIXED" ] && ok "chowned to aiscuser:$FIXED" || skip "already aiscuser-owned"

echo "== 4. mounts =="
for m in /mnt/data /mnt/output; do
  if [ -d "$m" ] && [ -n "$(ls -A $m 2>/dev/null)" ]; then ok "$m mounted"; else skip "$m not mounted (no storage: block in the yaml)"; fi
done

echo "== 5. VS Code machine settings =="
# Two purposes. (a) TERMINAL USER: a VS Code terminal is root with HOME=/home/aiscuser, so
# anything run in it writes root-owned files into aiscuser's home and repo -- model weights,
# __pycache__ and .vscode/settings.json were all found root-owned on 2026-09-14. Defaulting the
# terminal profile to `su - aiscuser` closes that at the source. (b) RESPONSIVENESS: the tunnel
# is ~2.8 MB/s via a relay in another region, so file watching and Pylance indexing over it are
# the difference between usable and not.
VSC=/home/aiscuser/.vscode-server/data/Machine
if [ -f "$VSC/settings.json" ]; then
  skip "$VSC/settings.json already present"
else
  mkdir -p "$VSC"
  cat > "$VSC/settings.json" <<'JSON'
{
  "terminal.integrated.profiles.linux": {
    "aiscuser": { "path": "su", "args": ["-", "aiscuser"] }
  },
  "terminal.integrated.defaultProfile.linux": "aiscuser",
  "files.watcherExclude": {
    "**/.git/objects/**": true, "**/__pycache__/**": true,
    "/scratch/omni-eval/**": true, "/opt/conda/**": true, "**/*.jsonl": true
  },
  "search.exclude": { "**/__pycache__": true, "/opt/conda": true },
  "search.followSymlinks": false,
  "python.analysis.indexing": false,
  "python.analysis.diagnosticMode": "openFilesOnly",
  "git.autorefresh": false,
  "git.autofetch": false,
  "extensions.autoUpdate": false
}
JSON
  ok "wrote $VSC/settings.json (new terminals open as aiscuser)"
fi

echo "== 6. $ENV_NAME env =="
if [ -n "${SKIP_ENV:-}" ]; then skip "SKIP_ENV set -- env is the caller's business"; else
if [ -x "$PY" ]; then
  skip "$ENV already exists"
else
  [ "$(id -u)" -eq 0 ] || { warn "must be root to create the env -- rerun as root"; exit 1; }
  if [ -n "$ENV_BASE" ]; then
    echo "  cloning $ENV_BASE -> $ENV_NAME (hardlinked)..."
    $CONDA create --clone "$ENV_BASE" -n "$ENV_NAME" -y -q > "/tmp/${ENV_NAME}_create.log" 2>&1 \
      && ok "cloned from $ENV_BASE" || { warn "clone failed -- see /tmp/${ENV_NAME}_create.log"; exit 1; }
  else
    echo "  creating a CLEAN python=3.10 env (not a clone)..."
    $CONDA create -n "$ENV_NAME" python=3.10 -y -q > "/tmp/${ENV_NAME}_create.log" 2>&1 \
      && ok "created" || { warn "create failed -- see /tmp/${ENV_NAME}_create.log"; exit 1; }
  fi
fi
# aiscuser must own it, so every later install is a plain user install
if [ "$(stat -c %U "$ENV")" = "aiscuser" ]; then
  skip "already owned by aiscuser"
else
  chown -R aiscuser:aiscuser "$ENV" && ok "chowned to aiscuser"
fi
fi   # SKIP_ENV

fi   # end root-side steps

# ---- steps 7-10 must run as aiscuser --------------------------------------------------------
if [ -z "$STAGE2" ] && [ "$(id -u)" -eq 0 ]; then
  echo "== re-running steps 7-10 as aiscuser =="
  # `su aiscuser -c` (no dash): a LOGIN shell as root is replaced by /etc/profile's
  # `exec su - aiscuser`, which waits on stdin and hangs a non-interactive caller.
  su aiscuser -c "ENV_NAME='$ENV_NAME' ENV_BASE='$ENV_BASE' REPO_URL='$REPO_URL' SKIP_ENV='${SKIP_ENV:-}' bash '$0' --as-aiscuser"
  RC=$?
  echo
  if [ -n "${QUIET_MANUAL:-}" ]; then exit $RC; fi
  echo "== manual, needs you =="
  echo "  1. W&B key:  cp $REPO/ch07/01_main-chapter-code/.env.example \\"
  echo "                  $REPO/ch07/01_main-chapter-code/.env    # then fill WANDB_API_KEY"
  echo "  2. Claude:   installed by step 10 -- just log in:"
  echo "               su - aiscuser && cd $REPO && claude"
  echo
  echo "  In VS Code, select the interpreter: $PY"
  exit $RC
fi

export PYTHONNOUSERSITE=1

echo "== 7. repo =="
if [ -z "$REPO_URL" ]; then
  skip "REPO_URL empty -- clone and editable install left to you"
elif [ -d "$REPO/.git" ]; then
  skip "$REPO already cloned"
else
  cd "$CODE" && git clone --quiet "$REPO_URL" && ok "cloned"
fi
if [ -n "$REPO_URL" ] && cd "$REPO" 2>/dev/null; then
  git config user.name  "Abhishek Goswami"
  git config user.email "abgoswam@gmail.com"
  ok "git identity set"
fi

echo "== 8. packages into $ENV_NAME =="
if [ -n "${SKIP_ENV:-}" ]; then
  skip "SKIP_ENV set"
elif [ -z "$REPO_URL" ]; then
  skip "no repo -- install into $ENV_NAME yourself"
  $PY -m pip install --quiet ipykernel 2>&1 | tail -1
elif $PY -c "import reasoning_from_scratch, wandb, transformers, ipykernel, torch" 2>/dev/null; then
  skip "already importable from $ENV_NAME"
else
  echo "  installing the repo and its deps (pulls torch + CUDA libs, ~3 GB on a fresh env)..."
  $PY -m pip install --quiet -e . 2>&1 | tail -2
  $PY -m pip install --quiet $PKGS 2>&1 | tail -2
  if $PY -c "import reasoning_from_scratch, wandb, transformers, ipykernel, torch" 2>/dev/null; then
    ok "installed into $ENV"
    $PY -c "import torch; print(f'        torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}')"
  else
    warn "imports still failing -- run: $PY -m pip check"
  fi
fi

echo "== 9. jupyter kernel =="
if [ -n "${SKIP_ENV:-}" ]; then skip "SKIP_ENV set -- register the kernel yourself"; else
# VS Code usually discovers a conda env holding ipykernel on its own, but the only kernelspec
# the image registers points at `ptca`, so an explicit one removes the ambiguity. `--user`
# writes to $HOME/.local/share/jupyter/kernels, and the VS Code server runs with
# HOME=/home/aiscuser, which is where it looks.
#
# The VS Code EXTENSIONS (ms-python.python, ms-toolsai.jupyter) cannot be automated FROM HERE:
# the VS Code server owns ~/.vscode-server/extensions and installs from what the CLIENT asks
# for, so a shell writing files there registers nothing. They are lost with the container like
# everything else. Automate it client-side instead, once, in the Windows user settings.json:
#     "remote.SSH.defaultExtensions": ["ms-python.python", "ms-toolsai.jupyter"]
# That reinstalls them on every connection, so a restart wipe is self-healing. Without it the
# kernel registered below exists but nothing on the node can open a notebook to use it.
if $PY -m jupyter kernelspec list 2>/dev/null | grep -q "^  $ENV_NAME "; then
  skip "kernel '$ENV_NAME' already registered"
else
  $PY -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" \
    > /dev/null 2>&1 && ok "registered kernel 'Python ($ENV_NAME)'" || warn "kernel registration failed"
fi
fi   # SKIP_ENV

echo "== 10. claude =="
# Only the LOGIN is interactive; the install is not. Doing it here, in the aiscuser stage, is
# what stops `curl | bash` from a root VS Code terminal leaving a root-owned ~/.claude that
# Claude itself cannot then write to.
if [ -x "$HOME/.local/bin/claude" ]; then
  skip "claude already at $HOME/.local/bin/claude"
else
  curl -fsSL https://claude.ai/install.sh 2>/dev/null | bash > /tmp/claude_install.log 2>&1 \
    && ok "installed -- run 'claude' and log in" || warn "install failed -- see /tmp/claude_install.log"
fi
