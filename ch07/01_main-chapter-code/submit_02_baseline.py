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
    evaluate_math500_stream,
    render_prompt,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
    load_model_and_tokenizer,
    load_math500_test,
    load_tokenizer_only,
)
from reasoning_from_scratch.ch04 import top_p_filter
from reasoning_from_scratch.ch06 import (
    load_math_train,
)
from reasoning_from_scratch.qwen3 import KVCache, Qwen3Model, QWEN_CONFIG_06_B

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


def wandb_log(payload, step):
    if WANDB_RUN is not None:
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

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

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
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
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


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
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
            example = math_data[step % len(math_data)]
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
                    num_correct, num_examples, acc = evaluate_math500_stream(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        math_data=subset,
                        out_path=out_path,
                        max_new_tokens=max_new_tokens,
                        verbose=False,
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

    # WANDB -- open the run FIRST. Any failure here aborts before the model
    # loads, so a run never trains with its metrics going nowhere.
    if args.no_wandb:
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

    if CSV_LOG_PATH.exists():
        raise SystemExit(
            f"{CSV_LOG_PATH} already exists.\n"
            "Refusing to append to a previous run's metrics -- pass a different "
            "--out_dir, or omit it to get a fresh timestamped directory."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing run outputs to: {OUT_DIR}")

    if WANDB_RUN is not None:
        (OUT_DIR / "wandb.txt").write_text(
            f"{WANDB_RUN.url}\n{WANDB_RUN.id}\n", encoding="utf-8"
        )
        WANDB_RUN.config.update({"out_dir": str(OUT_DIR)}, allow_val_change=True)

    if args.seed is not None and str(args.seed).strip().lower() != "none":
        torch.manual_seed(int(args.seed))
    device = torch.device(args.device) if args.device else get_device()

    math_data = load_math_train()
    if args.checkpoint_path:
        tokenizer = load_tokenizer_only(which_model="base")
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model="base", device=device, use_compile=False
        )

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

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), CHECKPOINT_DIR/"qwen3-0.6B-rlvr-grpo.pth")

    # WANDB -- flush and close the run
    if WANDB_RUN is not None:
        WANDB_RUN.finish()

