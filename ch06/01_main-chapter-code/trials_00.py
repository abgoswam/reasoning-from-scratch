# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Local editable mirror of reasoning_from_scratch/ch06.py.
# Every trials_XX_*.py imports from here, so this is the one file to edit
# when experimenting. Check for drift with:
#   diff <(sed 1,7d trials_00.py) ../../reasoning_from_scratch/ch06.py

from reasoning_from_scratch.qwen3 import KVCache
from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer, render_prompt
)
from reasoning_from_scratch.ch04 import top_p_filter

import json
import os
import subprocess
import time
from pathlib import Path

import requests
import torch


def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        "MATH/math_full_minus_math500.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            print("Using backup URL.")
            r = requests.get(backup_url, timeout=30)
            r.raise_for_status()

        data = r.json()

        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


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
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)

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


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # Require \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        # Stage 1: generate rollouts
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Stage 2: compute rewards
        reward = reward_rlvr(text, example["answer"])

        # Stage 4: compute logprobs
        logp = sequence_logprob(model, token_ids, prompt_len)

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    # Stage 2: collect all rewards
    rewards = torch.tensor(roll_rewards, device=device)

    # Stage 3: compute advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # Stage 4: collect all logprobs
    logps = torch.stack(roll_logps)

    # Stage 5: compute policy gradient loss
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss  # In the next chapter we add a KL term here

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def append_csv_metrics(
    csv_log_path,
    step_idx,
    total_steps,
    loss,
    reward_avg,
    avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,
    mem_debug=False,
):
    if steps is None:
        steps = len(math_data)

    # Stage 1: initialize optimize
    # (the model was already initialized outside the function)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0

    if mem_debug:
        cuda_mem("after model+optimizer", device)

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)
    try:
        # Stage 2: Iterate over training steps
        for step in range(steps):

            # Stage 3: Reset loss gradient
            # (it's best practice to do this at the beginning of each step)
            optimizer.zero_grad()

            current_step = step + 1
            example = math_data[step % len(math_data)]

            if mem_debug:
                torch.cuda.reset_peak_memory_stats(device)
                print(f"  --- step {current_step} ---")

            # Stage 4: calculate GRPO loss
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            if mem_debug:
                cuda_mem("after rollouts + loss", device)

            # Stage 5: Backward pass to calculate loss gradients
            stats["loss_tensor"].backward()

            if mem_debug:
                cuda_mem("after backward", device)

            # Clip large gradients to improve training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Stage 6: Update model weights using loss gradients
            optimizer.step()

            if mem_debug:
                cuda_mem("after optimizer.step", device)

            # Stage 7: Collect rewards, losses, and response lengths
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path,
                current_step,
                steps,
                stats["loss"],
                reward_avg,
                avg_response_len,
            )

            # Print step metrics
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )

            # Sample outputs (every 10 steps) to check if model
            # generates coherent text
            if current_step % 10 == 0:
                print(f"[Step {current_step}] sample outputs")
                for i, sample in enumerate(stats["samples"][:3]):
                    text = sample["text"].replace("\n", "\\n")
                    print(
                        f"  {i+1}) reward={sample['reward']:.3f} "
                        f"len={sample['gen_len']}: {text}"
                    )
                print()

            # Stage 8: Save model checkpoint
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

    # Save a model checkpoint if we interrupt the training early
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


# ---------------------------------------------------------------------------
# Notebook-only definitions: these appear in ch06_main.ipynb but not in
# reasoning_from_scratch/ch06.py, because the chapter builds them up in stages.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1: end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.mean(next_token_logps).item()


def sequence_logprob_draft(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    start = prompt_len - 1
    end = token_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=token_ids.device)
    next_tokens = token_ids[start + 1: end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.sum(next_token_logps)


# ---------------------------------------------------------------------------
# Shared setup used by the trials_XX_*.py scripts.
# ---------------------------------------------------------------------------

RAW_PROMPT = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)

ROLLOUTS = [
    r"\boxed{83}",
    r"The correct answer is \boxed{83}",
    r"The final answer is 83",
    r"We get \boxed{38}",
]


def load_base_model(device=None):
    from reasoning_from_scratch.ch03 import load_model_and_tokenizer

    if device is None:
        device = torch.device("cpu")
    model, tokenizer = load_model_and_tokenizer(
        which_model="base", device=device, use_compile=False
    )
    return model, tokenizer, device


def pick_cuda_device():
    """Returns the CUDA device with the most free memory.

    get_device() returns a bare "cuda", which is always cuda:0. On a shared
    box where another job owns cuda:0, that is the one device to avoid.

    Free memory is read via nvidia-smi rather than torch.cuda.mem_get_info,
    which would initialize a ~260 MiB CUDA context on every device it probes.
    """
    from reasoning_from_scratch.ch02 import get_device

    if not torch.cuda.is_available():
        return get_device()

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        free_mib = [int(line) for line in out.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return get_device()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        order = [int(i) for i in visible.split(",") if i.strip()]
        free_mib = [free_mib[i] for i in order if i < len(free_mib)]

    free, idx = max((m, i) for i, m in enumerate(free_mib))
    print(f"Using cuda:{idx} ({free / 1024:.1f} GiB free)")
    return torch.device(f"cuda:{idx}")


def cuda_mem(tag, device=None):
    if device is None or torch.device(device).type != "cuda":
        return
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    free = torch.cuda.mem_get_info(device)[0] / 1024 ** 3
    print(
        f"  [mem] {tag:<24} allocated={alloc:5.2f}  "
        f"peak={peak:5.2f}  free_on_gpu={free:5.2f}  (GiB)"
    )
