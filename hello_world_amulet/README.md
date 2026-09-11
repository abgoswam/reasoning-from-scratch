# Hello world on Azure Singularity with Amulet

Getting a job onto the Azure H100 Singularity cluster, the way
`ch07/01_main-chapter-code/submit_job_cluster.sh` gets one onto bonete61.

This directory is a **complete, self-contained hello world**: `src/hello.py` prints where it
landed and writes a file to blob storage. That is enough to prove installation, Azure auth, code
upload, scheduling, logs, and result download all work — before any real training is involved.

The CLI is **`amlt`**, not `amulet`. The Amulet source and examples are cloned separately at
`../../amulet/`; this folder is unrelated to that checkout.

## Quick reference — interactive box

In WSL, from this directory:

```bash
./run.sh interactive-fast                       # note the job name it prints
amlt status agoswami-interactive                # wait for `running` (~5 min)
python vscode_ssh_config.py agoswami-interactive <job-name> \
    --windows --alias amlt-box --write          # always --windows for VS Code
```

In VS Code, open a **local window** (`File → New Window`, no remote indicator bottom-left)
→ **Remote-SSH: Connect to Host…** → `amlt-box`

Then in the VS Code terminal, on the node:

```bash
echo '[ -e /tmp/amlt-env ] && . /tmp/amlt-env' >> "$HOME/.bashrc"  # $HOME, not /root
. /tmp/amlt-env                                                   # conda + python
cd /scratch/amlt_code
nohup python keep_wake.py > outputs/keepalive.log 2>&1 &          # or the box pauses
```

Both blocks are **per-job** — redo after every resubmit; the ssh endpoint is minted
per submission. Full explanation: [Interactive box](#interactive-box--submit-ssh-and-connect-vs-code).

## Working directory matters

**Every `amlt` command below must be run from this directory.** Amulet is directory-bound: the
`.amltconfig` file here ties this folder to a cloud project, and `amlt` looks for it relative to
your current directory.

```bash
cd "$(git rev-parse --show-toplevel)/hello_world_amulet"
```

A consequence worth knowing: `.amltconfig` stores **absolute** paths. If you move or rename this
folder, `LOCAL_PATH` and `DEFAULT_OUTPUT_DIR` go stale — and `amlt project checkout` will *not*
fix them, because it short-circuits when the project is already active. Repair with:

```bash
amlt project set default-output-dir ./amlt
amlt project          # verify LOCAL_PATH and DEFAULT_OUTPUT_DIR
```

## Concepts

Four moving parts. Getting them straight makes every command below obvious.

| Concept | What it is | Ours |
| --- | --- | --- |
| **Project** | Storage account + container holding uploaded code, metadata, and results. Bound to a local directory by `.amltconfig`. | `agoswami-hello` in `aifrontierssadata/data` |
| **Workspace** | The Azure ML workspace owning the compute and the managed identity your job runs as. | `ai-frontiers-sa-ws` |
| **Target** | A Singularity virtual cluster (VC) in that workspace. | `ai-frontiers-sa-vc` (H100 only) |
| **Experiment** | One named submission. Jobs from a config run under one experiment name. | `agoswami-hello-world` |

## Current state of this machine

Steps 1–8 have **already been run** on `agoswami`'s box. Re-running them is harmless, but you do
not need to. On a fresh machine, run 1–6 in order; 7 and 8 are only needed if you want `amlt ssh`.

| # | Step | Status here |
| --- | --- | --- |
| 1 | Install `uv` + `amlt` | done — `amlt` 11.19.0 |
| 2 | `az login` | done — `sc-uk5762764@microsoft.com` |
| 3 | Select subscription | done — `ASG Azure ML` |
| 4 | Create project | done — `agoswami-hello`, created 2026-09-03 06:56 |
| 5 | Register workspace | done — `ai-frontiers-sa-ws` is the project default |
| 6 | Check out project here | done — `.amltconfig` binds this folder |
| 7 | `az extension add --name ml` | done — **only needed for `amlt ssh`**, see below |
| 8 | SSH keypair in WSL `~/.ssh/` | done — copied from Windows, same fingerprint |

Verify at once:

```bash
amlt --version
az account show --query "{subscription:name,user:user.name}" --output table
amlt project
az extension list --query "[].name" -o tsv      # expect 'ml' for amlt ssh
ls -l ~/.ssh/id_ed25519                          # expect mode 600
```

That last command should print `PROJECT_NAME agoswami-hello`, a `LOCAL_PATH` ending in
`/hello_world_amulet`, and `PROJECT_DEFAULT_WORKSPACE ai-frontiers-sa-ws@ai-frontiers-rg`.

## Step 1 — Install `uv` and `amlt`

Requires the **Microsoft VPN**. Installation, workspace discovery, and job submission all reach
internal Microsoft services.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv tool install amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo
amlt --version
```

`uv tool install` keeps `amlt` in its own isolated environment, so it cannot collide with your
conda env. It lands in `~/.local/bin`, which must be on your `PATH`.

On Windows, do all of this inside WSL (`wsl --install -d Ubuntu` from an elevated PowerShell).

## Step 2 — Log in to Azure

```bash
az login
```

## Step 3 — Select the subscription

```bash
az account set --subscription "ASG Azure ML"
az account show --query "{subscription:name,user:user.name}" --output table
```

This changes your **global** `az` default, affecting every other terminal. You need **Storage Blob
Data Contributor** and **Storage Table Data Contributor** on `aifrontierssadata`, where Amulet
keeps project metadata and code uploads.

## Step 4 — Create the project

Check what exists before creating anything, so you don't end up with two projects holding half
your experiments each:

```bash
amlt project list aifrontierssadata data
```

If your project is not listed, create it — from this directory, because `create` also checks out:

```bash
amlt project create "<your-alias>-hello" aifrontierssadata data --output-dir ./amlt
```

## Step 5 — Register the workspace

```bash
amlt workspace add ai-frontiers-sa-ws --subscription "ASG Azure ML" --yes
amlt workspace set-project-default ai-frontiers-sa-ws
```

Confirm the target exists and see how busy it is:

```bash
amlt target list manifold
amlt target info -t ai-frontiers-sa-vc
```

## Step 6 — Check out the project here

Only needed if the project already exists and this folder is not yet bound to it (no
`.amltconfig`, or you cloned fresh — `.amltconfig` is gitignored because its paths are
machine-specific):

```bash
amlt project checkout agoswami-hello aifrontierssadata data
amlt project set default-config hello_world.yaml
amlt project
```

## Step 7 — Skip the image vulnerability gate

Amulet refuses to submit when the Docker image has no current FedRAMP scan record. **No image in
`aifrontierssacr` currently has one** — not even the production training image the phitrain
recipe uses — so every submission needs:

```bash
export FEDRAMP_SCANNER_SCAN_MODE=none
```

Without it, `amlt run` prints `No vulnerability scan found for image ...` and exits **without
creating the experiment**. Verify with `amlt list` — if the run was blocked, you will see no
experiment at all.

This is Amulet's documented escape hatch and it disables a real security check. It is defensible
only because these are team-owned images already running on this VC. The durable fix is an
`amlt scan-image --scan-mode trivy+fedramp` step in the image build pipeline.

## Step 8 — Submit

```bash
amlt run hello_world.yaml :hello-world agoswami-hello-world \
  --description "First amulet hello world" \
  --sla Premium \
  --yes
```

`:hello-world` is the job selector. **Always pass one** — a bare `amlt run hello_world.yaml`
submits every job in the file. Avoid `--replace` unless you mean to overwrite an experiment of
the same name.

On success Amulet prints the code upload, the auto-detected identity
(`_AZUREML_SINGULARITY_JOB_UAI: ai-frontiers-id`), and a status table with an `aka.ms/amlt?q=...`
portal link.

## Step 9 — Monitor

```bash
amlt status agoswami-hello-world :hello-world
amlt show   agoswami-hello-world :hello-world
amlt logs view -n 200 agoswami-hello-world :hello-world
amlt logs list        agoswami-hello-world :hello-world
```

Status moves `preparing` → `queued` → `running` → `pass`.

**Expect a long `queued`.** `ai-frontiers-sa-vc` has a single Premium quota pool that is routinely
fully booked by the phitrain training jobs — `amlt target info -t ai-frontiers-sa-vc` shows
used/limit, and at 224/224 nothing new starts until a running job releases GPUs. A queued job is
healthy; the cluster is just full.

Cancel with:

```bash
amlt cancel agoswami-hello-world :hello-world
```

## Step 10 — Collect results

`src/hello.py` writes `hello.txt` to `$AMLT_OUTPUT_DIR`, which Amulet maps into blob storage:

```bash
amlt results list     agoswami-hello-world :hello-world
amlt results download agoswami-hello-world :hello-world --output ./amlt-results
```

## Anatomy of `hello_world.yaml`

```yaml
target:
  service: manifold          # Singularity; "amlt target list manifold" lists targets
  name: ai-frontiers-sa-vc
  workspace_name: ai-frontiers-sa-ws

environment:
  registry: aifrontierssacr.azurecr.io
  image: nvidia25.11-...     # a *.azurecr.io image the workspace identity can pull

code:
  local_dir: $CONFIG_DIR/src # uploaded on every run; $CONFIG_DIR is this file's directory

jobs:
  - name: hello-world
    priority: high
    sku: 80G1-H100           # 80GB, 1 H100 — "amlt cache instance-type -s NDH100v5"
    command:
      - python hello.py --out_dir "$$AMLT_OUTPUT_DIR"
```

Three rules behind most first-time failures:

- **`$` vs `$$`.** A single `$VAR` is expanded **locally, at submission time**; `$$VAR` reaches
  the remote shell. Cluster variables — `AMLT_OUTPUT_DIR`, `RANK` — always need `$$`.
- **`code.local_dir` must exist and be non-empty.** Commands run from its remote copy, which is
  why `python hello.py` resolves: `hello.py` sits at the root of `src/`.
- **Write outputs only to `$$AMLT_OUTPUT_DIR`.** Anything else vanishes when the node is released.

`sku` reads as `<GPU-mem>G<count>-<accelerator>` with an optional `<nodes>x` prefix. `80G1-H100`
is the smallest slice this VC sells; `2xG8-H100` in the phitrain recipe is two 8-GPU nodes.

The image choice is deliberate: a small `python3.12-slim-pybox` image would start faster, but it
is blocked by the same scan gate, and the large NVIDIA image is already cached on these nodes and
carries the torch/CUDA stack the real training job will need.

## Mapping from `submit_job_cluster.sh`

| bonete61 | Amulet |
| --- | --- |
| `--upload "$REPO_ROOT"` | `code.local_dir` |
| `--node 1 --gpu-per-node 1` | `sku: 80G1-H100` |
| `--cmd "$JOB_CMD"` | `jobs[].command` |
| `$OUTPUT_DIR` | `$$AMLT_OUTPUT_DIR` |
| `JOB_NAME` | experiment name + `jobs[].name` |
| `PRIORITY=p0` | `priority: high` + `--sla Premium` |
| `set -a; . ./.env` | `submit_args.env` |

Secrets: never put values in the YAML. Reference them as
`submit_args.env: { WANDB_API_KEY: $WANDB_API_KEY }` (single `$` — resolved locally) and export
them in your shell before `amlt run`. Amulet stores the resolved values in job submission
metadata, so use team-approved credentials rather than personal long-lived tokens, and never run
`amlt run --dump` with such a config — it prints them.

## Reference

- Amulet docs: [aka.ms/amulet](https://aka.ms/amulet) · config reference:
  <https://amulet-docs.azurewebsites.net/config_file.html>
- Amulet source and examples: `../../amulet/`
  (`examples/mnist_pytorch/manifold_simple.yaml` is the canonical minimal config)
- A real multi-node training launcher: `../../phitrain/recipes/rl/tool_agent/amulet/`

## The config files

Two configs remain. The cluster/image matrix below was verified in 2026-09; the
`baltic*.yaml` and `interactive.yaml` variants were removed on 2026-09-11 once their
lesson was recorded, so the middle rows are a record, not files you can run.

| File | Cluster | Image | Verified |
| --- | --- | --- | --- |
| `hello_world.yaml` | ai-frontiers-sa-vc | team ACR, in-region | passes — 3 min run |
| *(removed)* `baltic.yaml` | baltic01 | `amlt-sing/acpt-torch2.8.x` platform | passed — ~5 min queue |
| *(removed)* `baltic_acr_slim.yaml` | baltic01 | `python3.12-slim-pybox` from our ACR | passed — ~10 min queue |
| *(removed)* `baltic_acr.yaml` | baltic01 | team ACR, multi-GB | never completed testing |
| *(removed)* `interactive.yaml` | baltic01 | team ACR | superseded by the `_fast` variant |
| `interactive_fast.yaml` | baltic01 | `amlt-sing/acpt-torch2.8.x` platform | passes — ssh verified 2026-09-04, again 2026-09-11 |

Submit any of them the same way:

```bash
export FEDRAMP_SCANNER_SCAN_MODE=none
amlt run <file>.yaml :<job> <experiment> --sla Premium --yes
```

Reuse one experiment name to get them side by side in a single `amlt status`.

## Interactive box — submit, ssh, and connect VS Code

`interactive_fast.yaml` holds a single H100 open so you can work on it directly.
Use `./run.sh`, which supplies the flags that are easy to forget:

```bash
./run.sh interactive-fast
```

That runs `amlt run interactive_fast.yaml :interactive-fast=<job>-<timestamp>
agoswami-interactive -i --sla Premium --yes`, and appends the job to `RUNS.md`.
**Note the job name it prints** — every step below needs it.

```bash
DRY_RUN=1 ./run.sh interactive-fast    # print the command, submit nothing
```

### Step A — wait for `running`

```bash
amlt status agoswami-interactive
```

About 5 minutes with the platform image.

### Step B — ssh in and START THE KEEPALIVE BY HAND

**This is the step that is easy to miss, and skipping it is why earlier boxes
paused after ~21 hours.**

`amlt ssh` requires the job to have been submitted with `-i`. But `-i` makes
Amulet replace the config's `command:` block with its own tmux harness — so the
keepalive declared in `interactive_fast.yaml` **never runs**. The GPU sits idle
and the platform eventually pauses the job.

So, every time you submit a box:

```bash
amlt ssh agoswami-interactive :<job-name>
```

then, on the node:

```bash
cd /scratch/amlt_code
nohup python keep_wake.py > outputs/keepalive.log 2>&1 &
nvidia-smi          # expect a python process holding ~620 MiB
```

`keep_wake.py` is copied verbatim from `e5-mistral-ft-code`, where it holds
Singularity jobs open for days: 180s of matmul on every GPU, then 600s rest.
A `0%` reading is normal — you caught the rest phase. Memory stays allocated.

**Caveat — the keepalive may not be what prevents pausing.** `/tmp/amlt-env` on
this job shows `SINGULARITY_ENABLE_PROCESS_ACTIVITY_MONITOR="false"`, i.e. idle
detection is *disabled* (the companion
`SINGULARITY_PROCESS_ACTIVITY_MONITOR_IDLE_DURATION_IN_SECONDS="1800"` is unarmed).
If that held for the boxes that paused at 21h and 23h, idle-reaping was never the
mechanism, and the real cause is more likely the previously-unset
`max_run_duration_seconds` or plain preemption. Keep the keepalive — it is cheap
and the setting may vary by cluster or SLA tier — but treat this as unresolved
until a box survives past ~24h.

### Step C — add the VS Code entry

This writes a `Host` block into `C:\Users\<you>\.ssh\config`, so VS Code can
connect by name. **Always use `--windows`:**

```bash
python vscode_ssh_config.py agoswami-interactive <job-name> \
    --windows --alias amlt-box --write
```

Then in VS Code: **Remote-SSH: Connect to Host…** → `amlt-box`.

**Open a *local* VS Code window first** — `File → New Window`, with no remote
indicator in the bottom-left. VS Code cannot nest one remote inside another, so
Remote-SSH is unavailable from a window already attached to WSL. If Remote-SSH
seems missing from the command palette, that is why.

Verify without opening VS Code:

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes amlt-box "hostname"
```

`node-0` and exit 0 means VS Code will work.

The script's default (no `--windows`) writes WSL's `~/.ssh/config` instead. That
is **not** useful for VS Code — it only serves plain `ssh amlt-box` from a WSL
terminal, which `amlt ssh <exp> :<job>` already does with no config at all.

### Step D — set up the shell inside VS Code

A VS Code terminal is a plain login shell, so **conda and `python` are missing**:

```
root@node-0:~# which python
root@node-0:~#          # nothing
```

`amlt ssh` doesn't have this problem because it sends an explicit remote command
that sources `/tmp/amlt-env` and `cd`s to the code dir before handing you a shell
(visible in `ssh -v` output as `debug1: Sending command: ...`). Nothing does that
for VS Code.

`/tmp/amlt-env` is the job's full environment — `PATH` with
`/opt/conda/envs/ptca/bin`, `AMLT_OUTPUT_DIR`, `AMLT_CODE_DIR`, and the rest.
Run these in the VS Code terminal, **in this order**:

```bash
echo '[ -e /tmp/amlt-env ] && . /tmp/amlt-env' >> "$HOME/.bashrc"  # future terminals
. /tmp/amlt-env                                                    # this terminal
cd /scratch/amlt_code
which python        # expect /opt/conda/envs/ptca/bin/python
conda env list      # expect ptca active
```

**Use `"$HOME/.bashrc"`, never a hardcoded path.** There are three plausible home
directories on one node and the shell picks a non-obvious one:

| Path | Whose |
| --- | --- |
| `/root` | what `whoami` (root) suggests — **wrong** |
| `/home/aiscuser` | the job's execution user, and what `/tmp/amlt-env` sets `HOME` to |
| `/home/azureuser` | **what `$HOME` actually is** — the ssh `User` from the Host block |

Symptom of getting it wrong: the line is present in the file you edited, a new
shell still has no `python`, and `echo $AMLT_JOB_NAME` is empty — proving the file
was never sourced.

Two more quirks of sourcing that file:

- It sets `HOME=/home/aiscuser`, so appending with `~` **after** sourcing writes to
  a different file than before. Append first, source second.
- It sets `PWD=/scratch/amlt_code`, which overwrites bash's own `PWD` — your prompt
  will claim you moved without `cd`-ing. The shell's real directory is unchanged;
  run `cd .` to resync.

For the Python extension, set the interpreter by hand — it does not see a `PATH`
that only exists after `.bashrc` runs:
**Python: Select Interpreter** → `/opt/conda/envs/ptca/bin/python`

Do not commit or paste `/tmp/amlt-env` anywhere: it contains a container SSH
private key, the AzureML run token, an MLflow token, and SAS URLs. All are
job-scoped and expire with the job, but they are real credentials.

Everything in Steps B and D is **per-job** — the node is rebuilt on every
submission, so the keepalive and the shell setup are redone each time.

### Keep `--alias` stable, and regenerate after every resubmit

The block's `HostName` is a `wss://` websocket URL minted **per submission**, so
it dies with the job. Re-run the Step C command after each resubmit. Keeping
`--alias amlt-box` fixed means VS Code's saved host keeps working — the script
replaces the block in place using its `# >>> amlt amlt-box >>>` markers.

### Why the Windows block differs

Amulet's block is WSL-bound: transport is a `ProxyCommand` running
`/opt/az/bin/python3` against the `az ml` extension under `~/.azure/`, neither of
which exists on Windows. `--windows` rewrites three fields:

| Field | Windows value | Why |
| --- | --- | --- |
| `ProxyCommand` | prefixed `wsl.exe -e` | delegates only the tunnel to WSL, reusing its `az login` |
| `IdentityFile` | `C:\Users\<you>\.ssh\id_ed25519` | Windows ssh cannot read `/home/...` |
| `UserKnownHostsFile` | `NUL` | Windows equivalent of `/dev/null` |

Copying the block to Windows *without* those rewrites cannot work, no matter how
the keys are arranged — `HostName` is a URL, not a host, so everything depends on
the ProxyCommand binary being reachable.

### If ssh fails

`vscode_ssh_config.py` reports **"The job must be running AND have been submitted
with `amlt run -i`"** for *every* failure — it only knows it could not find an ssh
command in Amulet's output. That message named the wrong cause in both real
failures we hit. Get the actual error:

```bash
python vscode_ssh_config.py agoswami-interactive <job-name> --dump /tmp/sshraw.txt
sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' /tmp/sshraw.txt | grep -iE "error|could not|failed"
```

Causes seen so far:

| Real error in the dump | Fix |
| --- | --- |
| `Could not find the Azure CLI SSH connector` | `az extension add --name ml` |
| `No authentication method succeeded` | `az login` (then re-pin `az account set --subscription "ASG Azure ML"`) |
| no key offered | copy `id_ed25519` into WSL `~/.ssh/` and `chmod 600` |

Amulet uses **your** key — it logs `Providing ~/.ssh/id_ed25519.pub … for ssh
login` at submit time. On WSL the keys often live only on the Windows side:

```bash
cp /mnt/c/Users/<you>/.ssh/id_ed25519* ~/.ssh/
chmod 600 ~/.ssh/id_ed25519
```

Copy, don't symlink — DrvFs (`/mnt/c`) cannot hold `0600`, and ssh rejects a
world-readable private key.

### Teardown

```bash
amlt cancel agoswami-interactive :<job-name>
```


## Choosing a cluster and an image

`ai-frontiers-sa-vc` is H100-only and frequently at 224/224, so queues are long. `baltic01` is a
shared GCR H100 cluster with 96 GPUs, usually mostly free. Both are driven by the same
`ai-frontiers-sa-ws` workspace, the same `ai-frontiers-id` identity and the same
`aifrontierssadata` storage — none of that has to change to move between them.

**The one thing that changes is the image.** `aifrontierssacr` sits in `ASG Azure ML` /
`southafricanorth`; `baltic01` nodes sit in `Singularity Shared` / `eastus2`. Pulling across that
gap is permitted — `baltic_acr_slim.yaml` proved it, with no extra role assignments — but a
multi-GB CUDA image has to travel first, and that time is charged to your queue.

So: prefer a **Manifold platform image** for a far cluster.

```bash
amlt cache base-images
```

Platform images are short aliases with no `registry:` field, resolved per-target by Singularity.
The trade-off is the stack they ship: `amlt-sing/acpt-torch2.8.x` carries torch 2.8, while the team
image carries torch 2.10.

**A long queue on a cluster with free quota usually means image staging, not scheduling.** Check
`amlt target info -t <vc>` first. Singularity reports staging as `queued`, and the status can flap
`queued → running → queued` while it settles.

## Two dead ends, recorded so they are not retried

**Do not use `-t baltic01` as a CLI override.** `-t` replaces only `target.name`;
`target.workspace_name` stays behind, so half the target comes from the file and half from the
command line. Keep them together in one config.

**You do not need the `gcrllm` stack.** There is a matched set there (`gcrllm2ws`, `gcrllm2cr`,
`GCRLLMUAI`, `gcrllm2data`), and `amlt workspace add gcrllm2ws` succeeds — but submitting through it
fails with `Microsoft.MachineLearningServices/workspaces/datastores/write` denied, needing the
**AzureML Data Scientist** role. Do not request that role; `ai-frontiers-sa-ws` already drives
`baltic01`. Note that `amlt workspace add` succeeding proves only `Reader` — registration does not
imply submission rights.
