# Fresh-node setup checklist

Everything below is redone on **every** new interactive box: the container banner
is explicit that local changes (installs, dotfiles, clones) are lost on restart,
failover, or resubmit. Written 2026-09-13 after losing half a day to step 5.

Submitting the box, waiting for `running`, the keepalive, and the VS Code SSH entry
are covered in [`hello_world_amulet/README.md`](../../hello_world_amulet/README.md);
this file starts once you have a VS Code terminal on the node.

## The one thing to understand first

There are two users and three kinds of shell on the node, and they behave differently:

| Shell | User | `$HOME` | Reads `/etc/profile`? | Result |
| --- | --- | --- | --- | --- |
| VS Code terminal (interactive, **non-login**) | root | `/home/aiscuser` | no | works; `.bashrc` supplies conda/python |
| `su - aiscuser`, `amlt ssh` (interactive, login) | aiscuser | `/home/aiscuser` | yes | works; banner printed |
| **Claude Code's Bash tool as root** (`bash -l -c`, non-interactive login) | root | `/home/aiscuser` | yes | **hangs forever** |

`/etc/profile` ends with a platform-injected block (present twice):

```bash
if [ "$(id -u)" -eq 0 ] || [ "$(id -un)" = "root" ]; then
  echo "Switching to aiscuser..."
  exec su - aiscuser
fi
```

Any login shell started as root is *replaced* by an interactive aiscuser shell that
waits on stdin. Claude Code launched from a root terminal therefore has a Bash tool
that never returns, not even for `echo ok`. The fix is not to edit `/etc/profile`
(the permission classifier refuses, and it would be lost anyway) but to **run Claude
Code as aiscuser**, where `id -u` is not 0 and the block never fires.

Consequence: anything created from the root terminal (the git clone, the Claude
install, `~/.claude`) is root-owned, and aiscuser then can't write to it and git
refuses with "dubious ownership". So do the setup as aiscuser from the start.

## Steps

### 1. Root terminal: conda/python for interactive shells

VS Code opens a root non-login shell with no `python`. Append to `.bashrc` — note
`$HOME` is already `/home/aiscuser` even as root, so this lands in aiscuser's file,
which is what you want:

```bash
cat >> "$HOME/.bashrc" <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
# job env (conda, AMLT_*), minus the cwd variables -- see note below
[ -e /tmp/amlt-env ] && . <(grep -Ev '^declare -x (PWD|OLDPWD|SHLVL)=' /tmp/amlt-env)
export PATH="$HOME/.local/bin:$PATH"
EOF
. "$HOME/.bashrc"
which python        # /opt/conda/envs/ptca/bin/python
```

`/tmp/amlt-env` is a `declare -x` dump of the job's environment. It contains real
credentials (container SSH key, run tokens, SAS URLs) — never cat, paste, or commit it.

**Why the `grep -v`.** The dump includes `declare -x PWD="/scratch/amlt_code"`.
Sourced as-is, that assigns the `$PWD` variable without running `cd`, so the
prompt (drawn from `$PWD`) says `/scratch/amlt_code` while the shell's real cwd is
still `$HOME` — `ls` then lists `hostfile`, `samples`, `azureml_job_env.sh` and it
looks like Amulet uploaded nothing. Filtering `PWD`/`OLDPWD`/`SHLVL` avoids it.
If you sourced the unfiltered file, `pwd -P` shows the truth and `cd .` resyncs.
`HOME=/home/aiscuser` is deliberately kept: it's what makes root's `~` land in the
same place as aiscuser's, so dotfiles and installs are shared between the two.

### 2. Root terminal: make login shells read `.bashrc`

`/home/aiscuser/.profile` is platform-generated and only prints the "Common Runtime"
banner; it does **not** source `.bashrc`, so `su - aiscuser` gets a bare PATH and
`claude` will be "command not found" later. Append:

```bash
cat >> /home/aiscuser/.profile <<'EOF'

# source .bashrc so login shells (su -, ssh) get the same PATH as VS Code terminals
if [ -n "$BASH" ] && [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi
EOF
```

`.bashrc` returns immediately for non-interactive shells, so this is safe for
Claude's Bash tool.

### 3. Root terminal: keepalive (per the amulet README)

```bash
cd /scratch/amlt_code
nohup python keep_wake.py > keepalive.log 2>&1 &
```

Check it's alive with `tail -2 /scratch/amlt_code/keepalive.log` (expect
"Resting for: 600 seconds" lines with a recent mtime), **not** with `ps`: `/proc`
is mounted `hidepid=invisible`, so from an aiscuser shell root's processes don't
exist, and vice versa. On 2026-09-13 the yaml's own keepalive log at
`$AMLT_OUTPUT_DIR/keepalive.log` was absent — the manual one is what holds the box.

### 4. Switch to aiscuser for everything else

```bash
su - aiscuser
```

Ignore the two `SINGULARITY_CHECKPOINT_RESTORE_STRATEGY_TYPE` / `declare` errors and
the banner — they come from malformed lines in `/etc/profile.d/singularity_base_profile.sh`
and are harmless. Check `whoami` says `aiscuser` and `which python` still resolves.

### 5. aiscuser: clone the repo

Amulet uploads only `code.local_dir` from the job yaml — for the interactive box
that is `hello_world_amulet/src/`, i.e. `hello.py`, `keep_wake.py`, and a
`repo.diff` — into `$AMLT_CODE_DIR=/scratch/amlt_code`. The full repo never comes
from Amulet; it is always a manual clone next to those files. `/scratch/amlt_code`
itself is world-writable (`drwxrwxrwx`), so aiscuser can clone there directly:

```bash
cd /scratch/amlt_code
git clone https://github.com/abgoswam/reasoning-from-scratch.git
cd reasoning-from-scratch
git config user.name  "Abhishek Goswami"
git config user.email "abgoswam@gmail.com"
```

Pushing over https needs a GitHub PAT (no credential helper, no `gh`, no `~/.netrc`
on the node — untested as of 2026-09-13). `git config --global credential.helper store`
then one push stores it in `~/.git-credentials` for the life of the node.

If the clone already exists but is root-owned (the symptom is git saying
"detected dubious ownership"), fix it from the root terminal instead of re-cloning:

```bash
chown -R aiscuser:aiscuser /scratch/amlt_code/reasoning-from-scratch
```

### 6. aiscuser: install the package and the extras `ptca` lacks

```bash
cd /scratch/amlt_code/reasoning-from-scratch
python -m pip install --user -e . --no-deps --quiet
python -m pip install --user --quiet tokenizers requests wandb sympy transformers
python -c "import reasoning_from_scratch, wandb, transformers; print('ok')"
```

`--user` keeps it out of the shared conda env; `~/.local/bin` is on PATH from step 1.
`transformers` is only needed for the `submit_02_baseline_phitrain*` variants.

### 7. aiscuser: W&B credentials

```bash
cp ch07/01_main-chapter-code/.env.example ch07/01_main-chapter-code/.env
# fill in WANDB_API_KEY (personal account, api.wandb.ai). .env is gitignored.
```

The scripts load `.env` themselves and let it override the shell, because the box
exports a `WANDB_API_KEY` for the work instance.

### 8. aiscuser: install Claude Code and launch it

```bash
curl -fsSL https://claude.ai/install.sh | bash     # -> ~/.local/bin/claude
which claude                                        # /home/aiscuser/.local/bin/claude
cd /scratch/amlt_code/reasoning-from-scratch
claude
```

First launch prompts for login again — `~/.claude` (auth, settings, project memory)
does not survive the node. Verify the Bash tool is alive before doing anything else:
ask for `echo ok; whoami; which python` and expect `ok / aiscuser / …/ptca/bin/python`
within a second. If it hangs, Claude was started as root — exit and redo step 4.

If the previous node's Claude install was done as root, `~/.claude` and
`~/.local/share/claude` may be root-owned; `chown -R aiscuser:aiscuser` both.

### 9. Optional sanity run

```bash
cd ch07/01_main-chapter-code
python submit_02_baseline.py --steps 3 --num_rollouts 2 --max_new_tokens 64 --no_wandb
```

Downloads the Qwen3-0.6B base weights on first use (~1.2 GB) and writes to
`runs/<timestamp>-local/`.

## Teardown reminder

`amlt cancel agoswami-interactive :<alias>` from WSL. Push anything you want to keep
first — the clone, `.env`, `runs/`, and `~/.claude` all go with the node.
