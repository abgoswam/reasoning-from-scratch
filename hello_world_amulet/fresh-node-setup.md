# Fresh-node setup checklist

Everything below is redone on **every** new interactive box: the container banner
is explicit that local changes (installs, dotfiles, clones) are lost on restart,
failover, or resubmit. Written 2026-09-13 after losing half a day to step 5.

**A container RESTART is a fresh node.** Observed 2026-09-14 on a 3-day-old box: every
`.bashrc` edit, the VS Code `Machine/settings.json`, and everything installed were gone,
and `/scratch/amlt_code` was back to the uploaded `src/` alone. One thing differs from a
resubmit: the ssh endpoint still works, so the same alias reconnects and it feels like a
dropped connection rather than a wipe. **The keepalive does NOT come back** -- `-i`
discarded the yaml's `command:` block at submit, and a restart does not reinstate it, so
the box sits with an idle GPU until you start it by hand. Redo steps 1-8.

## Which box are you setting up?

Two tracks. They differ in the image, where packages live, and which repo lands on the node;
everything from *Terminals, and staying out of root* onward is shared.

| | **Track A — basic** | **Track B — phitrain** |
| --- | --- | --- |
| config | `interactive_fast.yaml` | `interactive_fast_phitrain.yaml` |
| image | platform PTCA | team ACR (`nvidia25.11-pytorch2.10.0-...`) |
| conda? | yes — `/opt/conda/envs/ptca` | **no conda at all**; python is `/usr/bin/python3` 3.12.3 |
| packages go to | a fresh conda env (`ptca3`, `ptca4`, …) | a venv at `/scratch/venvs/phitrain` (`--system-site-packages`) |
| repo | `reasoning-from-scratch` (public, cloned by the script) | `microsoft/aifsdk` (**private** — needs a token) |
| you run | `fresh_node_setup.sh` on the node | `bootstrap_phitrain_node.sh` from the laptop |
| steps | A–E, then the script | four commands, fully scripted |

Track A is the default box for reasoning-from-scratch work. Track B exists because phitrain
needs torch/vllm/flash-attn built against the team image; rebuilding those with pip is not a
fight worth having, so that track inherits them instead of installing them.

Adding a third repo later: if it is public and pip-installable, Track A with `REPO_URL=` set is
enough. If it is private or needs the team image, copy Track B — the driver script is the part
that generalises.

## Track A (basic) — before you have a terminal (from WSL)

### A. Submit

```bash
cd /home/abgoswam/_hackerreborn/aifsdk/reasoning-from-scratch/hello_world_amulet
./run.sh interactive-fast
```

`run.sh:37` already passes `-i` -- do not add it again; the dry run shows it twice and it
looks like a bug. `-i` is REQUIRED: `vscode_ssh_config.py` shells out to `amlt ssh` and
scrapes the `ssh command:` line, and that endpoint only exists for an interactive job.
The cost is that the yaml's `command:` block is discarded, so the keepalive and any
staging are manual (steps 3 and 10 below).

Edit `interactive_fast.yaml` FIRST if this box needs anything the last one did not --
`sku`, or a `storage:` block. Mounts and GPU count are fixed at submit time; a running
box cannot gain them.

### B. Wait for `running`

```bash
amlt status agoswami-interactive
```

`amlt ssh` fails until then, and reports the same misleading message it gives for every
other cause. An 8-GPU box gang-schedules, so it queues longer than a 1-GPU one; a changed
config also means a fresh code package and image staging (~30 min observed).

### C. Generate the VS Code ssh entry

```bash
python vscode_ssh_config.py agoswami-interactive <full-job-name> \
  --windows --alias amlt-box-<MMDD> --write
```

`--windows` is mandatory: VS Code is a Windows process, so it always uses `ssh.exe` and
`C:\Users\<you>\.ssh\config` no matter where the job was submitted from. Without it the
block lands in WSL's `~/.ssh/config` and only serves `ssh` from a WSL terminal.
The endpoint is minted per SUBMISSION, so this is re-run after every resubmit (but not
after a mere restart).

### D. Verify ssh before involving the editor

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes amlt-box-<MMDD> "hostname"
```

If this fails, **do not trust the error text** -- `vscode_ssh_config.py` prints "must be
running AND submitted with `amlt run -i`" for ANY failure, including a stale `az login` or
a missing `az extension add --name ml`. Re-run step C with `--dump /tmp/ssh.log` and read
the real error.

### E. Connect

VS Code -> Remote-SSH: Connect to Host... -> `amlt-box-<MMDD>`, from a **local** window
(File -> New Window). VS Code cannot nest remotes, so Remote-SSH is unavailable from a
window already attached via Remote-WSL -- the symptom is the command appearing to be
missing from the palette.

Then continue below, from a VS Code terminal on the node.

## Both tracks: the one thing to understand first

There are two users and three kinds of shell on the node, and they behave differently:

| Shell | User | `$HOME` | Reads `/etc/profile`? | Result |
| --- | --- | --- | --- | --- |
| VS Code terminal (interactive, **non-login**) | root | `/home/aiscuser` | no | works; `.bashrc` supplies conda/python |
| `su - aiscuser`, `amlt ssh` (interactive, login) | aiscuser | `/home/aiscuser` | yes | works; banner printed |
| `ssh host "cmd"` (non-interactive, non-login) | root | **`/home/azureuser`** | no | works, but a DIFFERENT `$HOME` -- see below |
| **Claude Code's Bash tool as root** (`bash -l -c`, non-interactive login) | root | `/home/aiscuser` | yes | **hangs forever** |

**`$HOME` is not constant.** It is `/home/aiscuser` in a VS Code terminal but `/home/azureuser`
over plain ssh (observed 2026-09-14), while `whoami` is `root` in both. Anything `$HOME`-relative
therefore lands in different places depending on how you got there -- which is exactly why a
`pip install --user` appeared to work in VS Code and failed over ssh, and why the script writes
to `$HOME/.bashrc` rather than a hardcoded path.

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
refuses with "dubious ownership".

The setup script handles this for you -- run it as root and it re-invokes itself with
`su aiscuser -c` (no dash, so `/etc/profile` never fires) for the steps that create files.
Verified 2026-09-14: zero root-owned files in either the env or the repo afterwards. Only
`claude` still has to be installed and launched as aiscuser by hand.

## Track A (basic) — setup: run the script

```bash
bash /scratch/amlt_code/fresh_node_setup.sh        # as root
```

`hello_world_amulet/src/fresh_node_setup.sh`, uploaded with every box because `src/` is the
job's `code.local_dir`, so it is already on the node. Idempotent -- re-run it after a restart,
or any time you are unsure. It does:

Override the env name to build a second one side by side:

```bash
ENV_NAME=ptca4 bash /scratch/amlt_code/fresh_node_setup.sh
```

| # | step | runs as | skipped when |
| --- | --- | --- | --- |
| 1 | GPU keepalive | caller | GPU already holds >100 MiB |
| 2 | `$HOME/.bashrc` sources `/tmp/amlt-env` | caller | line present |
| 3 | `/home/aiscuser/.profile` sources `.bashrc` | caller | line present |
| 4 | report which mounts exist | caller | never (read-only) |
| 5 | VS Code machine settings — terminal profile + perf | root | file present |
| 6 | create the clean env, chown to aiscuser | root | env exists |
| 7 | clone the repo, set git identity | aiscuser | `.git` exists |
| 8 | `pip install -e .` + extras into the env | aiscuser | imports work |
| 9 | register the jupyter kernel | aiscuser | kernel listed |
| 10 | install `claude` | aiscuser | binary present |

It runs as root, then **re-executes itself** as aiscuser (`su aiscuser -c`, no dash) for steps
7-10, so everything those steps create is aiscuser-owned. Verified 2026-09-14: zero root-owned
files in the env, and `~/.claude` owned by aiscuser.

One thing is left to you, printed at the end:

```bash
cp ch07/01_main-chapter-code/.env.example ch07/01_main-chapter-code/.env   # fill WANDB_API_KEY
```

and then `claude` (installed by step 10) needs its interactive login:
`su - aiscuser && cd $REPO && claude`. Verify its Bash tool before doing anything else — ask for
`echo ok; whoami; which python` and expect an answer within a second. If it hangs, Claude was
started as root; exit and `su - aiscuser` first.

In VS Code, select the interpreter `/opt/conda/envs/<env>/bin/python`, and **reload the window**
so step 5's settings take effect.

## Track B (phitrain) — four commands from WSL

The whole track is scripted. You type four things, all in WSL, all from
`reasoning-from-scratch/hello_world_amulet`:

| # | command | notes |
| --- | --- | --- |
| 1 | `CONFIG=interactive_fast_phitrain.yaml ./run.sh interactive-fast-phitrain` | the `CONFIG=` prefix is **required** — `run.sh` defaults to `interactive_fast.yaml`, a different box |
| 2 | `amlt status :interactive-fast-phitrain` | wait for `running` |
| 3 | `python vscode_ssh_config.py --windows <full-job-name> --alias amlt-phitrain-<MMDD> --write` | endpoint is minted per submission; re-run after every resubmit |
| 4 | `./bootstrap_phitrain_node.sh amlt-phitrain-<MMDD>` | everything else |

Step 4 is the whole rest of the setup: node config, GitHub token, sparse clone, editable
install, verification. It is idempotent — re-run it after a container restart.

**You do not run `fresh_node_setup_phitrain.sh` yourself.** It ships to the node via
`code.local_dir: src` and the driver invokes it over ssh as step 3 of its own sequence. It has a
guard that refuses to run anywhere that is not an Amulet node, so a stray local invocation
cannot chown your laptop's home directory.

### What the driver does

| # | where | as | step |
| --- | --- | --- | --- |
| 2 | laptop | you | preflight: gh account is `abgoswam` **and** `microsoft/aifsdk` is visible |
| 3 | node | root | `bash /scratch/amlt_code/fresh_node_setup_phitrain.sh` |
| 4 | laptop→node | — | `gh auth token` piped into `/home/aiscuser/.git-token` (0600) |
| 5 | node | aiscuser | sparse clone `aifsdk`, then strip the token from `.git/config` |
| 6 | node | aiscuser | `pip install -e phiagent -e phitrain --no-deps` + pins |
| 7 | node | aiscuser | verify imports resolve to `/scratch/amlt_code` |

### The four things that bit, 2026-09-14

**The gh account, not SSO.** A token from an account without `microsoft` org access clones with
`remote: Repository not found` — a 404 standing in for a 403, because GitHub hides private repos
you cannot see. It is not a bad token and not a typo. `agoswami_microsoft` fails, `abgoswam`
works. The driver's preflight now checks both the account name and actual repo visibility before
it touches the node, which turns a 30-minute detour into an immediate error.

**`--no-deps` is load-bearing.** Without it pip resolves phitrain's full dependency tree and
replaces the inherited torch and vllm — which is the entire reason this track exists.

**The pins step is easy to miss.** If it does not run, `pip list` shows `lightning 2.6.1` and
`omegaconf 2.0.0`. Those are the image's *system* versions showing through
`--system-site-packages`, not the pins, and they look plausible enough to skip over.

**`phitrain-cli: command not found`** after a successful install is a `PATH` problem, not a
broken install. The console script only ever exists at `/scratch/venvs/phitrain/bin/phitrain-cli`,
and calling the venv's python by full path does not put that directory on `PATH`. Either
`source /scratch/venvs/phitrain/bin/activate`, or use the module form
`/scratch/venvs/phitrain/bin/python -m phitrain.cli.interface` — which is what omni-eval itself
switched to in PR #3656, for exactly this reason.

### Why a venv and not `PYTHONUSERBASE`

`phitrain_baltic.yaml :smoke` uses `PYTHONUSERBASE=/tmp/amlt-user`. That works for a batch job,
but it is only visible to processes that have the variable set — fresh terminals and notebook
kernels do not, which is the same trap as `pip install --user`. The venv is visible to anything
that names its python, and VS Code can select it as an interpreter.

## Shared: applies to both tracks

From here down, nothing is track-specific except the one section marked Track A.

## Terminals, and staying out of root

A VS Code terminal is **root with `$HOME=/home/aiscuser`**, so anything you run in it writes
root-owned files into aiscuser's home and repo. Found root-owned on 2026-09-14 after ordinary
work: downloaded model weights under `ch06/`, `math_train.json`, `.vscode/settings.json`, and
every `__pycache__/*.pyc`. Nothing breaks immediately — git stays quiet — but aiscuser then
cannot overwrite them, and it surfaces later as a confusing permission error.

Step 5 sets the default terminal profile to `su - aiscuser`, so after a window reload new
terminals are aiscuser and the problem stops at the source. Before that reload, or in an old
terminal, run `su - aiscuser` yourself.

**Notebooks are not covered by this.** The Jupyter extension starts the kernel as the server
user (root), not through a terminal profile, so a notebook can still write root-owned files.
Check with `import os; os.getuid()` in a cell — `0` means root. To repair:

```bash
chown -R aiscuser:aiscuser /scratch/amlt_code/reasoning-from-scratch
```

## Track A (basic): why a separate clean env, not `ptca`

`ptca` is the image's env: root-owned and shared. Two approaches were tried and rejected on
2026-09-14:

- **`pip install --user`** -- visible only when `$HOME` is `/home/aiscuser`. Works in a VS Code
  terminal, fails over ssh with `ModuleNotFoundError`, because there `$HOME` is
  `/home/azureuser`. Silent and confusing.
- **installing into `ptca` itself** -- needs root (its `site-packages` is mode 777 but pip needs
  more than that, and the failure is hidden by `--quiet`), and root then leaves root-owned
  `egg-info` in your repo and root-owned files in the env that aiscuser cannot later change.

The env (default `ptca3`, override with `ENV_NAME`) is created clean
(`conda create -n <env> python=3.10`), chowned to aiscuser, and everything goes in as a plain
user install. Clean rather than a clone so the env holds exactly
what the repo declares -- the cost is pip pulling torch and its CUDA libs from PyPI (~3 GB, a
few minutes on first run). Verified on two independent builds (`ptca3`, `ptca4`): `torch 2.10.0+cu128, cuda=True`.

Step 9 registers a kernelspec (`~/.local/share/jupyter/kernels/<env>`) so the Jupyter extension
lists it unambiguously — the only kernel the image registers points at `ptca`. The VS Code
EXTENSIONS themselves (`ms-python.python`, `ms-toolsai.jupyter`) cannot be scripted: VS Code
installs them into `~/.vscode-server/extensions` on first connect, and they go with the
container like everything else, so expect that prompt again after a restart.

**Set `PYTHONNOUSERSITE=1` for any manual pip call.** If a user-site copy of a dependency
exists, pip counts it as satisfied and omits it from the env -- which is how one env ended up
with `wandb` but not `click`, then not `opentelemetry`, then not `regex`, each only surfacing
after the previous was fixed.

## Checking the keepalive

Use the **GPU**, never `pgrep`:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Memory held (~629 MiB per GPU) means a live CUDA context. `keep_wake.py` bursts 180s then rests
600s, so `0 %` WITH memory held is healthy mid-rest; `0 %, 0 MiB` is dead. The log corroborates:
`tail -2 /scratch/amlt_code/keepalive.log` plus its mtime.

`pgrep -f keep_wake` lies three ways: it matches your own command line (reporting a keepalive
that does not exist), `pkill -f` on the same pattern kills your own shell, and `/proc` is
`hidepid=invisible` so the other user's processes are hidden either way. To kill the loop, build
the pattern at runtime so it is absent from your command line:

```bash
P=$(printf "keep%s" _wake); for pid in $(pgrep -f "$P"); do [ "$pid" != "$$" ] && kill "$pid"; done
```

## Storage mounts and staging weights

Only if the box was submitted with a `storage:` block. `/mnt/data` is the `data` container,
`/mnt/output` is `aion-jobs`; `/mnt/data/X` is blob `X` verbatim. `/mnt/output` is mounted even
without a `storage:` block -- it is Amulet's own results mount.

Copy weights to `/scratch` before using them repeatedly: storage is in `southafricanorth`,
`baltic01` is in `eastus2`, so every read crosses regions, while `/scratch` is node-local NVMe
(28 TB). Time one shard first, then parallelise -- blobfuse single-stream is the bottleneck:

```bash
time cp /mnt/data/models/<model>/model-00001-of-000NN.safetensors /scratch/probe.bin
mkdir -p /scratch/<model>
ls /mnt/data/models/<model>/*.safetensors | xargs -P 8 -I{} cp {} /scratch/<model>/
```

## Optional sanity run

```bash
cd /scratch/amlt_code/reasoning-from-scratch/ch07/01_main-chapter-code
/opt/conda/envs/ptca3/bin/python submit_02_baseline.py --steps 3 --num_rollouts 2 \
  --max_new_tokens 64 --no_wandb
```

Downloads Qwen3-0.6B on first use (~1.2 GB) and writes to `runs/<timestamp>-local/`.

## Teardown reminder

`amlt cancel agoswami-interactive :<alias>` from WSL. Push anything you want to keep
first — the clone, `.env`, `runs/`, and `~/.claude` all go with the node.
