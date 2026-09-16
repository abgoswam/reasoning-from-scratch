#!/usr/bin/env bash
# Drive a fresh phitrain box from the LAPTOP, end to end.
#
#   ./bootstrap_phitrain_node.sh amlt-phitrain-0914
#   ./bootstrap_phitrain_node.sh amlt-phitrain-0914 --dry-run
#
# Runs steps 2-7 of the sequence documented in src/fresh_node_setup_phitrain.sh. Step 1
# (submitting the job) stays manual, and so does generating the ssh Host block:
#
#   CONFIG=interactive_fast_phitrain.yaml ./run.sh interactive-fast-phitrain
#   python vscode_ssh_config.py --windows <job-name>
#
# Everything after that is here. Safe to re-run: every step checks before it acts.
#
# The token is piped straight from `gh auth token` into ssh stdin. It is never echoed, never
# written to a local file, and never passed as an argument (so it cannot show up in ps).
set -uo pipefail

NODE="${1:-}"
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1
SSH="${SSH:-/mnt/c/Windows/System32/OpenSSH/ssh.exe}"
VENV=/scratch/venvs/phitrain
CODE=/scratch/amlt_code

die()  { printf '\033[31mFAIL\033[0m  %s\n' "$1" >&2; exit 1; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
step() { printf '\n== %s ==\n' "$1"; }

[ -n "$NODE" ] || die "usage: $0 <node-alias> [--dry-run]    (alias from vscode_ssh_config.py)"

# ssh.exe is chatty on stderr; drop the known noise, keep anything real.
rssh() { "$SSH" -o BatchMode=yes "$NODE" "$@" 2>&1 \
         | grep -vE "Deprecation|websockets|Read stream|Detected non-Windows|Warning: Perm"; }

# --- step 2: laptop-side preflight ------------------------------------------------------------
step "2. laptop preflight"
command -v gh >/dev/null || die "gh not installed"
WHO="$(gh api user --jq .login 2>/dev/null)" || die "gh has no working token -- run: gh auth login"
[ "$WHO" = "abgoswam" ] || die "active gh account is '$WHO', need 'abgoswam'. Run: gh auth switch --user abgoswam
       ('$WHO' may lack microsoft org access, which shows up later as a bogus 404 on clone.)"
ok "gh account $WHO"
gh api repos/microsoft/aifsdk --jq .full_name >/dev/null 2>&1 \
  || die "account $WHO cannot see microsoft/aifsdk (404 = no org access, not a bad token)"
ok "microsoft/aifsdk visible"
[ -x "$SSH" ] || die "ssh not found at $SSH (override with SSH=...)"

if [ "$DRY" = 1 ]; then
  echo; echo "DRY RUN -- would now, on $NODE:"
  echo "  3. bash $CODE/fresh_node_setup_phitrain.sh            (as root)"
  echo "  4. write /home/aiscuser/.git-token from 'gh auth token'"
  echo "  5. sparse clone microsoft/aifsdk -> $CODE/aifsdk      (as aiscuser)"
  echo "  6. pip install -e phiagent -e phitrain --no-deps + pins"
  echo "  7. verify imports"
  exit 0
fi

REMOTE_WHO="$(rssh 'whoami' | tr -d '\r' | tail -1)"
[ "$REMOTE_WHO" = "root" ] || die "ssh to $NODE gave user '$REMOTE_WHO', expected root (is the job still running?)"
ok "$NODE reachable as root"

# --- step 3: node setup, as root --------------------------------------------------------------
step "3. node setup (root)"
rssh "bash $CODE/fresh_node_setup_phitrain.sh" || die "fresh_node_setup_phitrain.sh failed"

# --- step 4: token ----------------------------------------------------------------------------
step "4. token"
gh auth token | "$SSH" -o BatchMode=yes "$NODE" \
  'cat > /home/aiscuser/.git-token && chown aiscuser:aiscuser /home/aiscuser/.git-token && chmod 600 /home/aiscuser/.git-token' \
  >/dev/null 2>&1 || die "could not write the token to the node"
SZ="$(rssh 'stat -c %s /home/aiscuser/.git-token' | tr -d '\r' | tail -1)"
[ "${SZ:-0}" -gt 20 ] || die "token file is $SZ bytes, that is wrong"
ok "token delivered ($SZ bytes, 0600, aiscuser)"

# --- steps 5-7: clone, install, verify, as aiscuser -------------------------------------------
step "5-7. clone, install, verify (aiscuser)"
"$SSH" -o BatchMode=yes "$NODE" "cat > /tmp/phitrain_user_setup.sh && chown aiscuser:aiscuser /tmp/phitrain_user_setup.sh && su aiscuser -c 'bash /tmp/phitrain_user_setup.sh'" 2>&1 <<INNER | grep -vE "Deprecation|websockets|Read stream|Detected non-Windows|Warning: Perm"
set -uo pipefail
VENV=$VENV
CODE=$CODE
ok()   { printf '  \033[32mok\033[0m    %s\n' "\$1"; }
skip() { printf '  --    %s\n' "\$1"; }

# -- 5a. credential helper --
# This is a --filter=blob:none partial clone, so ANY later checkout/fetch has to pull blobs
# from origin. Stripping the token out of .git/config (below) therefore breaks the very next
# 'git checkout <branch>' with an interactive "Username for 'https://github.com':" prompt.
# The helper keeps the token in one 0600 file instead of in every remote URL. Seen 2026-09-14.
if [ "\$(git config --global credential.helper)" = "store" ] && [ -s /home/aiscuser/.git-credentials ]; then
  skip "credential helper already configured"
else
  printf 'https://x-access-token:%s@github.com\n' "\$(cat /home/aiscuser/.git-token)" \
    > /home/aiscuser/.git-credentials
  chmod 600 /home/aiscuser/.git-credentials
  # 'store' is inert on its own -- without this line the file is never consulted.
  git config --global credential.helper store
  ok "credential helper: store -> ~/.git-credentials (0600)"
fi

# -- 5b. clone --
if [ -d "\$CODE/aifsdk/.git" ]; then
  skip "\$CODE/aifsdk already cloned"
else
  cd "\$CODE" || exit 1
  TOKEN=\$(cat /home/aiscuser/.git-token)
  # --filter=blob:none --sparse: aifsdk is a ~27 GB tree and we need two directories.
  git clone --quiet --filter=blob:none --sparse \
      "https://x-access-token:\$TOKEN@github.com/microsoft/aifsdk.git" || exit 1
  cd aifsdk || exit 1
  git sparse-checkout set phitrain phiagent || exit 1
  # strip the token back out of .git/config -- the helper above serves it from now on
  git remote set-url origin https://github.com/microsoft/aifsdk.git
  ok "cloned, sparse: phitrain phiagent"
fi
cd "\$CODE/aifsdk" || exit 1

# -- 6. install -- relative paths, so this MUST run from the repo root (we are there).
if "\$VENV/bin/python" -c 'import phitrain, phiagent' 2>/dev/null; then
  skip "phitrain + phiagent already installed"
else
  # --no-deps is load-bearing: without it pip resolves phitrain's full tree and replaces the
  # inherited torch and vllm from the image.
  "\$VENV/bin/python" -m pip install --quiet -e phiagent -e phitrain --no-deps || exit 1
  ok "editable install: phiagent, phitrain"
fi
# pins from phitrain_baltic.yaml :smoke -- the combination proven against this image
"\$VENV/bin/python" -m pip install --quiet \
  'apache-tvm-ffi<0.1.10' 'kernels==0.14.1' 'lightning==2.6.5' \
  'omegaconf==2.4.0.dev14' 'transformers==5.9.0' 'wandb==0.26.0' \
  sentencepiece tiktoken || exit 1
ok "pins installed"

# -- 7. verify --
"\$VENV/bin/python" - <<'PY' || exit 1
import torch, phitrain, phiagent
print(f"  ok    torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
print(f"  ok    phitrain from {phitrain.__file__}")
PY
[ -x "\$VENV/bin/phitrain-cli" ] && ok "phitrain-cli present (needs: source \$VENV/bin/activate)"
INNER

step "done"
cat <<EOF
  On the node:   su - aiscuser && source $VENV/bin/activate && phitrain-cli --help
  In VS Code:    interpreter $VENV/bin/python   |   kernel 'Python (phitrain)'

  git is configured to use the token for fetch/checkout/push (credential.helper=store).
  It lives in two 0600 files on the node. To revoke it early:
    $SSH $NODE 'rm /home/aiscuser/.git-token /home/aiscuser/.git-credentials'
EOF
