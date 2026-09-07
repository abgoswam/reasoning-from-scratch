# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Self-contained copy of ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py
# (the section 7.2 baseline run -- plain GRPO, no KL term), with Weights &
# Biases logging added. Everything the run needs is in this one file.
#
# The only changes from the original are marked "# WANDB" -- five sites:
#   1. the guarded import below
#   2. WANDB_RUN + wandb_log() helper
#   3. append_step_metrics   -> mirrors every CSV row
#   4. append_eval_metrics   -> mirrors MATH-500 accuracy
#   5. append_sample_logs    -> rollout text as a wandb Table
#   (+ argparse flags and wandb.init/finish in __main__)
#
# The eight changes marked "# PHITRAIN" swap the hand-written Qwen3Model for a
# Hugging Face module and then rewrite it in place the way phitrain does:
#   1. distributed setup -- torchrun ranks -> a device mesh
#   2. AutoModelForCausalLM   <- was Qwen3Model + torch.load(.pth)
#   3. apply_ac        = phitrain.models.parallelisms.activation_checkpoint
#   4. apply_fsdp      = phitrain.models.parallelisms.data_parallel
#   5. generation -> past_key_values instead of KVCache (sample_response + eval)
#   6. sequence_logprob -> .logits instead of a raw tensor
#   7. data-parallel prompts, one GRPO group per rank
#   8. peak-memory reporting, and rank-0 guards on every writer
#
# compute_grpo_loss and train_rlvr_grpo are untouched: the algorithm does not
# change when the weights are sharded across four GPUs.
#
# Check for drift against upstream with:
#   diff <(sed -n "/^import argparse/,$p" submit_02_baseline.py) \
#        <(sed -n "/^import argparse/,$p" \
#          ../../ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py)
#
# W&B is REQUIRED unless you pass --no_wandb. If wandb is missing, unauthenticated,
# or unreachable, the run aborts BEFORE the model loads rather than training blind.
# Every metric goes to both W&B and a fresh local directory per run:
#
#   runs/<timestamp>-<wandb-run-id>/  metrics.csv  metrics.txt  outputs.txt
#                                     wandb.txt    checkpoints/
#
# Run (the section 7.2 run from ch07_main.ipynb):
#   python submit_02_baseline.py --steps 500 --max_new_tokens 1024 \
#       --device cuda:3 --wandb_project rfs-ch07
#   python submit_02_baseline.py --steps 10 --no_wandb    # local-only smoke test

import argparse
import os
import time
from pathlib import Path

import torch

try:  # WANDB (1/5) -- optional, so --no_wandb works on a machine without it
    import wandb
except ImportError:
    wandb = None

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    render_prompt,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
    load_math500_test,
)
from reasoning_from_scratch.ch04 import top_p_filter
from reasoning_from_scratch.ch06 import (
    load_math_train,
)
# PHITRAIN (1/8) -- torch's own distributed primitives; these are what
# phitrain.models.parallelisms wraps. transformers replaces qwen3.Qwen3Model.
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

SCRIPT_NAME = Path(__file__).stem
RUNS_ROOT = Path(__file__).parent / "runs"

# Placeholders -- __main__ rebinds all five to a fresh per-run directory before
# training starts, so two runs can never append to the same metrics.csv.
OUT_DIR = RUNS_ROOT / "unset"
LOG_PATH = OUT_DIR / "outputs.txt"
METRICS_LOG_PATH = OUT_DIR / "metrics.txt"
CSV_LOG_PATH = OUT_DIR / "metrics.csv"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

# WANDB (2/5) -- set in __main__; stays None for --no_wandb or a missing wandb
WANDB_RUN = None

# PHITRAIN (1/8) -- rebound in __main__ from the torchrun environment.
RANK, WORLD_SIZE, MESH = 0, 1, None


def wandb_log(payload, step):
    # PHITRAIN (8/8) -- one writer only, or N ranks fight over the same run.
    if WANDB_RUN is not None and is_main():
        WANDB_RUN.log(payload, step=step)

def load_env_file(path=Path(__file__).parent / ".env"):
    """Load KEY=VALUE lines from .env into os.environ, overriding the shell.

    Overriding matters: this box exports a WANDB_API_KEY for the work instance,
    and .env is what points these runs at the personal one.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


# PHITRAIN (1/8) -- torchrun sets RANK/WORLD_SIZE/LOCAL_RANK. A device mesh names
# the axes to shard over; phitrain builds one with "tensor_parallel" and
# "data_context_parallel" axes. Only the data-parallel axis is needed here.
def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, None
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    return dist.get_rank(), world_size, mesh


def is_main():
    return RANK == 0


# PHITRAIN (3/8) -- = phitrain.models.parallelisms.activation_checkpoint.apply_ac
# Recompute each decoder layer's activations in the backward pass instead of
# storing them. phitrain locates the layer list by dotted path; so does this.
def apply_ac(model):
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i] = checkpoint_wrapper(layers[i], preserve_rng_state=False)


# PHITRAIN (4/8) -- = phitrain.models.parallelisms.data_parallel.apply_fsdp
# fully_shard() is FSDP2. Applied per decoder layer, each rank keeps 1/N of that
# layer's parameters and gathers the rest only while the layer runs. The last
# layer stays gathered because backward needs it immediately -- phitrain does
# the same via reshard_after_forward.
def apply_fsdp(model, mesh, param_dtype=torch.bfloat16):
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    layers = model.model.layers
    for i, layer in enumerate(layers):
        fully_shard(layer, mesh=mesh, mp_policy=policy,
                    reshard_after_forward=(i < len(layers) - 1))
    fully_shard(model, mesh=mesh, mp_policy=policy)


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
        )

    # PHITRAIN (5/8) -- the ch07 model owns a KVCache and reset_kv_cache().
    # An HF module exposes past_key_values instead: same algorithm, different
    # handle. This is the "HF defines what the layers are" boundary, concretely.
    cache = DynamicCache()
    logits = model(input_ids.unsqueeze(0), past_key_values=cache, use_cache=True).logits[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)

        # In the core chapters, we used .cpu() for better consistency across systems,
        # but it causes a 20% performance hit when training on GPUs
        # next_token = torch.multinomial(probas.cpu(), num_samples=1).to(device)
        next_token = torch.multinomial(probas, num_samples=1)

        token_id = next_token.item()
        generated.append(token_id)

        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, past_key_values=cache, use_cache=True).logits[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


def sequence_logprob(model, token_ids, prompt_len):
    # PHITRAIN (6/8) -- HF returns a ModelOutput, so logits come off .logits.
    # use_cache=False is set unconditionally: this is a single full-sequence
    # forward that never reads a cache, so building one is pure waste (though
    # measurably free at this size). It becomes REQUIRED with --activation-ckpt,
    # where the recomputed forward would build a different DynamicCache than the
    # original and torch.utils.checkpoint rejects the mismatch. HF's own
    # gradient_checkpointing_enable() disables use_cache for the same reason.
    logits = model(token_ids.unsqueeze(0), use_cache=False).logits.squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return selected[prompt_len - 1:].sum()


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # Require \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    skip_zero_adv=False,
):
    roll_rewards, samples, rollout_data = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        reward = reward_rlvr(text, example["answer"])

        roll_rewards.append(reward)
        rollout_data.append((token_ids, prompt_len))
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    is_zero_adv = torch.allclose(
        advantages,
        torch.zeros_like(advantages),
        atol=1e-8,
        rtol=0.0,
    )

    if skip_zero_adv and is_zero_adv:
        return {
            "loss": 0.0,
            "pg_loss": 0.0,
            "rewards": roll_rewards,
            "advantages": advantages.detach().cpu().tolist(),
            "is_zero_adv": True,
            "samples": samples,
            "loss_tensor": None,
        }

    roll_logps = []
    for token_ids, prompt_len in rollout_data:
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)

    logps = torch.stack(roll_logps)

    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "is_zero_adv": is_zero_adv,
        "samples": samples,
        "loss_tensor": loss,
    }


def append_sample_logs(step_idx, samples, max_samples=3):
    if not is_main():  # PHITRAIN (8/8)
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Step {step_idx}] sample outputs\n")
        for i, sample in enumerate(samples[:max_samples]):
            text = sample["text"].replace("\n", "\\n")
            f.write(
                f"  {i+1}) reward={sample['reward']:.3f} "
                f"len={sample['gen_len']}: {text}\n"
            )
        f.write("\n")

    # WANDB (5/5) -- rollout text is the one thing a metrics chart cannot show
    if wandb is not None and WANDB_RUN is not None:
        table = wandb.Table(columns=["step", "reward", "gen_len", "text"])
        for sample in samples[:max_samples]:
            table.add_data(
                step_idx, sample["reward"], sample["gen_len"], sample["text"]
            )
        wandb_log({"rollouts": table}, step=step_idx)


def append_step_metrics(
    step_idx,
    total_steps,
    loss,
    reward_avg,
    tokens_per_sec,
    avg_response_len,
    eval_acc=None,
):
    if not is_main():  # PHITRAIN (8/8)
        return
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Step {step_idx}/{total_steps}] "
            f"loss={loss:.4f} reward_avg={reward_avg:.3f} "
            f"tokens_per_sec={tokens_per_sec:.1f} "
            f"avg_response_len={avg_response_len:.1f}\n"
        )
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_LOG_PATH.exists():
        CSV_LOG_PATH.write_text(
            "step,total_steps,loss,reward_avg,tokens_per_sec,avg_response_len,eval_acc\n",
            encoding="utf-8",
        )
    with CSV_LOG_PATH.open("a", encoding="utf-8") as f:
        eval_acc_str = "" if eval_acc is None else f"{eval_acc:.6f}"
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{tokens_per_sec:.6f},{avg_response_len:.6f},{eval_acc_str}\n"
        )

    # WANDB (3/5) -- same numbers as the CSV row above
    wandb_log(
        {
            "loss": loss,
            "reward_avg": reward_avg,
            "tokens_per_sec": tokens_per_sec,
            "avg_response_len": avg_response_len,
        },
        step=step_idx,
    )


def append_eval_metrics(step_idx, acc, correct, total):
    if not is_main():  # PHITRAIN (8/8)
        return
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Eval step {step_idx}] math500_acc={acc:.4f} "
            f"({correct}/{total})\n"
        )

    # WANDB (4/5)
    wandb_log(
        {"eval_acc": acc, "eval_correct": correct, "eval_total": total},
        step=step_idx,
    )


# PHITRAIN (5/8) -- MATH-500 accuracy, rebuilt on sample_response so it works
# against an HF module. Near-greedy decoding (the original used a greedy
# helper), the same reward function as training, the same jsonl output.
@torch.no_grad()
def evaluate_math500_hf(model, tokenizer, device, math_data, out_path, max_new_tokens):
    import json

    num_correct = 0
    was_training = model.training
    model.eval()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in math_data:
            _, _, text = sample_response(
                model=model, tokenizer=tokenizer,
                prompt=render_prompt(row["problem"]), device=device,
                max_new_tokens=max_new_tokens, temperature=0.1, top_p=1.0,
            )
            correct = reward_rlvr(text, row["answer"]) > 0.0
            num_correct += int(correct)
            f.write(json.dumps({"problem": row["problem"], "generated": text,
                                "correct": bool(correct)}) + "\n")
    if was_training:
        model.train()
    total = len(math_data)
    return num_correct, total, (num_correct / total if total else 0.0)


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    if not is_main():  # PHITRAIN (8/8) -- see __main__ for the sharded-save note
        return None
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = checkpoint_dir / f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    math500_eval_data,
    device,
    steps=None,
    num_rollouts=9,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=CHECKPOINT_DIR,
    eval_max_items=0,
    skip_zero_advantage_updates=False,
    show_eta=False,
):
    if steps is None:
        steps = len(math_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    train_start_time = time.time() if show_eta else None
    try:
        for step in range(steps):
            step_start = time.perf_counter()
            current_step = step + 1
            # PHITRAIN (7/8) -- data parallelism: each rank takes a DIFFERENT
            # example and forms its own GRPO group; FSDP averages the gradients
            # across ranks. This is what `advantage_normalization: prompt`
            # means in the phitrain SWE config.
            example = math_data[(step * WORLD_SIZE + RANK) % len(math_data)]
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                skip_zero_adv=skip_zero_advantage_updates,
            )
            if stats["loss_tensor"] is not None:
                optimizer.zero_grad()
                stats["loss_tensor"].backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_time = time.perf_counter() - step_start
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            tokens_per_sec = step_tokens / step_time if step_time > 0 else 0.0
            if current_step % 10 == 0:
                append_sample_logs(current_step, stats["samples"])

            eval_acc = None
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")
                if eval_max_items and math500_eval_data:
                    was_training = model.training
                    model.eval()
                    subset = (
                        math500_eval_data[:eval_max_items]
                        if eval_max_items
                        else math500_eval_data
                    )
                    out_path = (
                        Path(checkpoint_dir)
                        / f"{SCRIPT_NAME}-step{current_step:05d}-math500.jsonl"
                    )
                    # PHITRAIN (5/8) -- evaluate_math500_stream() drives the
                    # ch07 model's own generate helper, so it cannot take an HF
                    # module. Same evaluation, rebuilt on sample_response.
                    num_correct, num_examples, acc = evaluate_math500_hf(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        math_data=subset,
                        out_path=out_path,
                        max_new_tokens=max_new_tokens,
                    )
                    eval_acc = acc
                    append_eval_metrics(current_step, acc, num_correct, num_examples)
                    print(
                        f"MATH-500 eval @ step {current_step}: "
                        f"acc={acc:.3f} ({num_correct}/{num_examples})"
                    )
                    if was_training:
                        model.train()

            append_step_metrics(
                current_step,
                steps,
                stats["loss"],
                reward_avg,
                tokens_per_sec,
                avg_response_len,
                eval_acc=eval_acc,
            )

            eta_suffix = ""
            if show_eta:
                eta_msg = eta_progress_message(
                    processed=current_step,
                    total=steps,
                    start_time=train_start_time,
                    show_eta=True,
                    label="Step",
                ).rstrip()
                eta_part = eta_msg.split(" | ", 1)[-1]
                eta_suffix = f" | {eta_part}"
            if is_main():  # PHITRAIN (8/8) -- one console writer
                print(
                    f"[Step {current_step}/{steps}] "
                    f"loss={stats['loss']:.4f} "
                    f"reward_avg={reward_avg:.3f} "
                    f"tok/sec={tokens_per_sec:.1f} "
                    f"avg_resp_len={avg_response_len:.1f}"
                    f"{eta_suffix}"
                )
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model


if __name__ == "__main__":
    load_env_file()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train RLVR GRPO on the MATH dataset."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=8,
        help="Number of rollouts per step.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per rollout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling cutoff.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Random seed (int) or None to disable seeding.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to resume training from.",
    )
    parser.add_argument(
        "--eval_on_checkpoint",
        type=int,
        default=0,
        help=(
            "Number of MATH-500 examples to evaluate at checkpoints "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--skip-zero-advantage-updates",
        action="store_true",
        help=(
            "Skip backward/optimizer step when rollout advantages are all "
            "near zero."
        ),
    )
    parser.add_argument(
        "--show_eta",
        action="store_true",
        help="Append ETA to step logs.",
    )
    # PHITRAIN -- flags below are additions for the parallelism passes
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HF model id or local HF-format directory.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Shard parameters across ranks with FSDP2 (needs torchrun).",
    )
    parser.add_argument(
        "--activation-ckpt",
        action="store_true",
        help="Recompute activations in backward instead of storing them.",
    )
    # WANDB -- flags below are additions to the upstream script
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rfs-ch07",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team). None uses your default.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Run name shown in W&B. None lets W&B generate one.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B and log to the local files only.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where metrics, samples, and checkpoints go. Defaults to a fresh "
             "./runs/<timestamp>-<wandb-run-id>/ directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:3. None auto-detects, which picks cuda:0 "
             "and will fail if that GPU is busy.",
    )
    args = parser.parse_args()

    # PHITRAIN (1/8) -- before anything touches a GPU or opens a W&B run.
    RANK, WORLD_SIZE, MESH = setup_distributed()

    # WANDB -- open the run FIRST. Any failure here aborts before the model
    # loads, so a run never trains with its metrics going nowhere.
    if args.no_wandb or not is_main():
        # PHITRAIN (8/8) -- only rank 0 opens a W&B run; others train silently.
        if args.no_wandb and is_main():
            print("W&B disabled (--no_wandb) -- local logs only")
        run_tag = "local"
    else:
        if wandb is None:
            raise SystemExit(
                "wandb is not installed, so this run cannot log to W&B.\n"
                "  Install it:  pip install wandb\n"
                "  Log in:      wandb login\n"
                "  Or run untracked:  --no_wandb"
            )
        WANDB_RUN = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                **vars(args),
                "stage": "7.2_baseline",
                "which_model": "base",
            },
        )
        run_tag = WANDB_RUN.id
        print(f"W&B run:    {WANDB_RUN.url}")
        print(f"W&B host:   {os.environ.get('WANDB_BASE_URL')}")
        print(f"W&B entity: {WANDB_RUN.entity}")

    # A fresh directory per run, tagged with the W&B id so the local files and
    # the dashboard point at each other.
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    else:
        OUT_DIR = RUNS_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_tag}"
    LOG_PATH = OUT_DIR / "outputs.txt"
    METRICS_LOG_PATH = OUT_DIR / "metrics.txt"
    CSV_LOG_PATH = OUT_DIR / "metrics.csv"
    CHECKPOINT_DIR = OUT_DIR / "checkpoints"

    # PHITRAIN (8/8) -- only rank 0 owns the run directory. Every rank computes
    # its own timestamp, so without this each would create a stray empty dir.
    if CSV_LOG_PATH.exists() and is_main():
        raise SystemExit(
            f"{CSV_LOG_PATH} already exists.\n"
            "Refusing to append to a previous run's metrics -- pass a different "
            "--out_dir, or omit it to get a fresh timestamped directory."
        )
    if is_main():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Writing run outputs to: {OUT_DIR}")

    if WANDB_RUN is not None:
        (OUT_DIR / "wandb.txt").write_text(
            f"{WANDB_RUN.url}\n{WANDB_RUN.id}\n", encoding="utf-8"
        )
        WANDB_RUN.config.update({"out_dir": str(OUT_DIR)}, allow_val_change=True)

    if args.seed is not None and str(args.seed).strip().lower() != "none":
        # PHITRAIN (7/8) -- offset by rank, or every rank draws identical
        # sampling noise for its own prompt. Rank 0 keeps the bare seed, so
        # single-GPU runs are numerically unchanged.
        torch.manual_seed(int(args.seed) + RANK)
    # PHITRAIN (1/8) -- under torchrun each rank owns exactly one GPU.
    if WORLD_SIZE > 1:
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    else:
        device = torch.device(args.device) if args.device else get_device()

    math_data = load_math_train()

    # PHITRAIN (2/8) -- the model-loading change. The original was:
    #     model = Qwen3Model(QWEN_CONFIG_06_B)                     # config hard-coded
    #     model.load_state_dict(torch.load("qwen3/...base.pth"))   # weights-only .pth
    # HF carries the architecture in config.json, so no config dict is needed.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))

    # phitrain's order: AC before FSDP, so the checkpoint wrapper ends up inside
    # the sharded unit rather than around it.
    if args.activation_ckpt:
        apply_ac(model)
    if args.fsdp and MESH is not None:
        apply_fsdp(model, MESH["dp"])
        if is_main():
            print(f"FSDP: params are DTensors, 1/{WORLD_SIZE} per rank")
    else:
        model.to(device)

    trained = train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=math_data,
        math500_eval_data=load_math500_test(),
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eval_max_items=args.eval_on_checkpoint,
        skip_zero_advantage_updates=args.skip_zero_advantage_updates,
        show_eta=args.show_eta,
    )

    # PHITRAIN (8/8) -- every rank reports; the spread shows sharding working.
    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[rank {RANK}] Max CUDA memory allocated: {max_mem_gb:.2f} GB", flush=True)

    # PHITRAIN (8/8) -- under FSDP each rank holds only a shard, so the state
    # dict must be gathered before saving. phitrain does this with
    # torch.distributed.checkpoint (StateDictOptions(full_state_dict=...));
    # this is the small single-file equivalent.
    if is_main():
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if WORLD_SIZE > 1:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )
        state = get_model_state_dict(
            trained, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    else:
        state = trained.state_dict()
    if is_main():
        torch.save(state, CHECKPOINT_DIR / "qwen3-0.6B-rlvr-grpo.pth")

    # WANDB -- flush and close the run
    if WANDB_RUN is not None:
        WANDB_RUN.finish()

    # PHITRAIN (1/8)
    if WORLD_SIZE > 1:
        dist.destroy_process_group()

