# `submit_02_baseline` and its phitrain variants

Three files, one algorithm. They isolate a single question: **what does a training
framework like phitrain add, and where does Hugging Face stop?**

| File | Model from | Runs on | Diff vs baseline |
| --- | --- | --- | --- |
| `submit_02_baseline.py` | hand-written `Qwen3Model` + `.pth` | 1 GPU | — |
| `submit_02_baseline_phitrain.py` | `AutoModelForCausalLM` | 1–N GPUs | 626 lines |
| `submit_02_baseline_phitrain_both.py` | `AutoModelForCausalLM` | 1–N GPUs, local or cluster | **167 lines** |

`_both` retains full parity with the baseline — W&B, MATH-500 eval, checkpoints,
`--out_dir`, every original flag. `_phitrain` is a stripped reading copy: 247 lines
against 852, at the cost of dropping everything except the mechanism.

## The eight changes

Marked `# PHITRAIN (n/8)` in both variants, following the convention the original
uses for its `# WANDB (n/5)` sites.

| # | Site | Change |
| --- | --- | --- |
| 1 | distributed setup | torchrun ranks → a `DeviceMesh` |
| 2 | model loading | `AutoModelForCausalLM` ⟵ `Qwen3Model` + `torch.load(.pth)` |
| 3 | `apply_ac` | activation checkpointing, per decoder layer |
| 4 | `apply_fsdp` | FSDP2 `fully_shard`, per decoder layer |
| 5 | generation | `past_key_values` ⟵ the ch07 `KVCache`; MATH-500 eval rebuilt |
| 6 | `sequence_logprob` | `.logits` ⟵ a raw tensor |
| 7 | data parallelism | each rank takes its own prompt, forms its own GRPO group |
| 8 | rank-0 guards | one writer for W&B, console, files, checkpoints |

**`compute_grpo_loss` and `train_rlvr_grpo` are untouched.** Group advantages and
the policy-gradient loss are identical whether the weights came from a `.pth` or
from Hugging Face, and whether or not they are sharded across four GPUs. Every
change above is plumbing.

### The parallelism passes

`apply_ac` and `apply_fsdp` are ~15 lines each, written inline against torch's own
primitives so every line is readable. Each is the small equivalent of a phitrain
module:

```
apply_ac    ≈ phitrain.models.parallelisms.activation_checkpoint
apply_fsdp  ≈ phitrain.models.parallelisms.data_parallel
```

Two details are copied from phitrain deliberately. **AC is applied before FSDP**,
so the checkpoint wrapper ends up inside the sharded unit rather than around it.
And the decoder layers are located by the dotted path `model.layers` — that path
is the entire interface phitrain depends on Hugging Face for.

## The two variants compared

They implement the same eight changes but were built differently, and the contrast
is instructive.

**`_phitrain.py` was written from scratch** — blank file, port only what the
demonstration needs, drop everything else. **`_both.py` was derived by patching** —
`cp submit_02_baseline.py`, then eight targeted replacements. Same conceptual
change, very different diffs:

| | `_phitrain.py` | `_both.py` |
| --- | --- | --- |
| built by | rewriting | patching the original |
| diff vs baseline | 626 lines | **167 lines** |
| total length | 247 lines | 852 lines |
| W&B, MATH-500 eval, checkpoints, `--out_dir` | absent | inherited |
| `--lr` | a flag | hard-coded `1e-5` |
| `--num_rollouts` / `--max_new_tokens` defaults | 4 / 128 | 8 / 512 |
| logprob computed | after all rollouts, in train mode | inside the loop, in eval mode |

The last row is the only difference in *training logic*. The baseline computes
each rollout's logprob inside the generation loop while `model.eval()` is active;
`_phitrain` generates all rollouts first, then computes logprobs after restoring
train mode.

### The results are identical

Same flags (`--steps 3 --num_rollouts 4 --max_new_tokens 320`), one GPU, seed 42:

| | step 3 loss | reward | peak memory |
| --- | --- | --- | --- |
| `_phitrain.py` | `+1.4325` | 0.500 | 6.64 GiB |
| `_both.py` | `1.4325` | 0.500 | 6.64 GiB |

Identical to four decimal places. That settles the structural question above: for
a model without dropout, `model.eval()` versus `model.train()` changes nothing
numerically, so the two orderings are equivalent. Matching losses across two
independently written implementations is also a reasonable check that neither
has a silent bug.

Both seed per rank (`seed + rank`), so rank 0 reproduces the single-GPU numbers
exactly while other ranks draw independent sampling noise.

### Which to use

**`_both`** for anything you intend to reproduce, compare, or run at length — it
has the full instrumentation and a diff that is readable as evidence.
**`_phitrain`** to follow the mechanism: 247 lines in one sitting, no W&B or
eval machinery in the way.

The general point: **when the question is "what does X change?", derive by
patching rather than rewriting.** A rewrite mixes the change under study with a
hundred incidental choices, and the diff stops being evidence. `git diff` on
`_both` shows exactly the eight things phitrain does.

## Running

```bash
PY=~/miniconda3/envs/pt_0811/bin/python
TR=~/miniconda3/envs/pt_0811/bin/torchrun

# one GPU
$PY submit_02_baseline_phitrain_both.py --steps 5 --no_wandb

# four GPUs, sharded
$TR --nproc_per_node=4 submit_02_baseline_phitrain_both.py \
    --steps 5 --num_rollouts 2 --max_new_tokens 64 \
    --fsdp --activation-ckpt --no_wandb

# with W&B (credentials from .env, rank 0 only)
$TR --nproc_per_node=4 submit_02_baseline_phitrain_both.py \
    --steps 500 --fsdp --activation-ckpt --wandb_project rfs-ch07
```

`--fsdp` is a no-op without `torchrun`, so one command line works either way.

### On the cluster

No code changes; add a job to `hello_rfs_amulet/train.yaml` with a multi-GPU sku.
Note `transformers` in the pip line — the baseline jobs do not need it.

```yaml
  - name: phitrain-fsdp
    priority: high
    sku: 80G8-H100
    command:
      - set -e -o pipefail
      - export PYTHONUSERBASE=/tmp/amlt-user
      - export PATH="$$PYTHONUSERBASE/bin:$$PATH"
      - python -m pip install --user -e . --no-deps --quiet
      - python -m pip install --user --quiet tokenizers requests wandb sympy transformers
      - cd ch07/01_main-chapter-code
      - mkdir -p "$$AMLT_OUTPUT_DIR"
      - torchrun --nproc_per_node=8 submit_02_baseline_phitrain_both.py
          --steps 500 --max_new_tokens 1024 --num_rollouts 8
          --fsdp --activation-ckpt --out_dir "$$AMLT_OUTPUT_DIR"
```

## Flags

Additions to the baseline's set:

| Flag | Default | Effect |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-0.6B` | HF model id or local HF-format directory |
| `--fsdp` | off | Shard parameters with FSDP2. Requires `torchrun` |
| `--activation-ckpt` | off | Recompute activations in backward |
| `--seed` | 42 | Seeds sampling; `_phitrain` seeds per rank (`seed + rank`) |

Everything else is inherited and unchanged: `--steps`, `--num_rollouts`,
`--max_new_tokens`, `--temperature`, `--top_p`, `--out_dir`, `--device`,
`--checkpoint_path`, `--eval_on_checkpoint`, `--show_eta`,
`--skip-zero-advantage-updates`, and the four W&B flags.

## Measurements

Peak CUDA memory, rank 0, Qwen3-0.6B, 256 new tokens, 4× RTX A6000:

| Configuration | Peak | vs baseline |
| --- | --- | --- |
| 1 rank, no flags | 5.60 GiB | — |
| 1 rank, `--activation-ckpt` | 5.58 GiB | ~0 |
| 4 ranks, `--fsdp` | 3.28 GiB | −41% |
| 4 ranks, `--fsdp --activation-ckpt` | 2.54 GiB | −55% |

**FSDP does the heavy lifting**, and the arithmetic says why. At 0.6B in bf16:

```
parameters       0.6B x 2 B      = 1.2 GiB
param.grad       0.6B x 2 B      = 1.2 GiB   } untouchable by
AdamW states     0.6B x 2 B x 2  = 2.4 GiB   } activation checkpointing
                                   --------
                                    4.8 GiB
measured peak                       5.58 GiB
                                   --------
activations                        ~0.8 GiB   <- all AC can ever save
```

About 86% of the footprint is parameters, their gradients and optimizer state.
Activation checkpointing cannot reduce any of it — it drops only the *transient*
saved tensors the autograd graph holds between forward and backward. That is why
`--activation-ckpt` alone moved 5.60 to 5.58 GiB, and why it became worth 0.74 GiB
once FSDP had sharded the other 4.8 GiB across four ranks. The flags are not
alternatives: FSDP has to work first before AC has much to bite on.

## Constraints

Five things that only surface once the model comes from `transformers` and the
parameters are sharded. Each is handled in the variants; the last column says
where, so the change can be inspected or reverted.

| Constraint | Symptom | Handled in |
| --- | --- | --- |
| Activation checkpointing and KV caching are incompatible | `CheckpointError: A different number of tensors was saved during the original forward and recomputation` — surfaces only with `--activation-ckpt` | `sequence_logprob()` — `use_cache=False`, set **unconditionally**.<br>`_phitrain.py:136` · `_both.py:216` |
| Under FSDP, `state_dict()` returns shards | A saved checkpoint silently contains 1/N of the weights | `__main__` — `get_model_state_dict(..., StateDictOptions(full_state_dict=True))`.<br>`_both.py:840` (`_phitrain` does not save) |
| Unseeded runs are not reproducible | Two runs, identical flags, completely different losses | `__main__` — `manual_seed(seed + rank)`.<br>`_phitrain.py:203` · `_both.py:778` |
| Every rank computes its own timestamp | Each rank creates a stray empty `runs/<timestamp>-local/` | `__main__` — run directory created under `is_main()`.<br>`_both.py:765` (`_phitrain` has no run directory) |
| `evaluate_math500_stream()` cannot take an HF model | It drives the ch07 model's own generate helper | Replaced by `evaluate_math500_hf()`, rebuilt on `sample_response`.<br>`_both.py:407` (`_phitrain` has no eval) |

Two of these deserve expanding.

**Why `use_cache=False`, and why unconditionally.** The model is called in exactly
two places, and only one of them wants a cache:

| Call site | Cache | Grad tracking | Reached by checkpointing |
| --- | --- | --- | --- |
| `sample_response()` | `past_key_values=cache, use_cache=True` | off — `@torch.no_grad()` | no |
| `sequence_logprob()` | **`use_cache=False`** | on — builds an autograd graph | yes |

`sample_response()` decodes token by token, so the cache is the whole point, and
running under `no_grad` means nothing is saved for backward and nothing is ever
recomputed. `sequence_logprob()` is a *single full-sequence* forward that never
reads a cache — building one is waste regardless of any other flag, which is why
the argument is not made conditional on `--activation-ckpt`.

Measured, the waste is negligible at this size: 5.59 GiB peak with or without the
argument when AC is off. The argument earns its place in the other case. With
`--activation-ckpt` the baseline's original call — which passed no `use_cache` at
all and so inherited `config.use_cache=True` for Qwen3 — builds a `DynamicCache`
on the first forward, a different one on recompute, and `torch.utils.checkpoint`
aborts on the saved-tensor mismatch. HF's own `gradient_checkpointing_enable()`
disables `use_cache` for exactly this reason.

**Why ranks must seed differently.** Each rank trains on its own prompt (change
7/8). Seeding every rank identically would make them draw the same sampling noise,
correlating the groups that GRPO is supposed to average over independently. Rank 0
keeps the bare seed, so single-GPU runs stay numerically identical to the table
above.

## Further reading

[`_reports/0904/same-loop-three-stacks.html`](../../_reports/0904/same-loop-three-stacks.html)
compares this loop against the phitrain SWE stack and a 397B SkyRL run; §3 traces
how each turns a checkpoint into a running model.
