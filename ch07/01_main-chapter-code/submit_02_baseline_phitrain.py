# Companion to submit_02_baseline.py: the same GRPO loop, but with the model
# loaded through Hugging Face `transformers` and then rewritten in place with
# the parallelism passes phitrain applies. Nothing else changes -- no vLLM, no
# Ray, no agent harness. The point is to isolate one seam.
#
# The eight changes from the original are marked "# PHITRAIN" below:
#   1. distributed setup (torchrun ranks -> a device mesh)
#   2. AutoModelForCausalLM instead of Qwen3Model + a .pth state dict
#   3. apply_ac      -- activation checkpointing, per decoder layer
#   4. apply_fsdp    -- FSDP2 fully_shard, per decoder layer
#   5. sample_response rewritten for HF's past_key_values
#   6. sequence_logprob rewritten for HF's .logits
#   7. data-parallel prompts: each rank takes its own example
#   8. peak-memory reporting, so the flags are measurable
#
# The GRPO maths (group advantages, policy-gradient loss) is untouched -- that
# is the whole finding: the algorithm does not change, only the plumbing.
#
# Run:
#   # 4 ranks, sharded
#   torchrun --nproc_per_node=4 submit_02_baseline_phitrain.py --steps 5 --fsdp --activation-ckpt
#   # 1 rank, for the memory comparison
#   python submit_02_baseline_phitrain.py --steps 5

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from reasoning_from_scratch.ch03 import (
    extract_final_candidate,
    grade_answer,
    render_prompt,
)
from reasoning_from_scratch.ch04 import top_p_filter
from reasoning_from_scratch.ch06 import load_math_train


# PHITRAIN (1/8) -- torchrun gives us RANK/WORLD_SIZE/LOCAL_RANK. A device mesh
# names the axes we shard over; phitrain builds one with "tensor_parallel" and
# "data_context_parallel" axes. We only need the data-parallel one.
def setup_distributed():
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, None

    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    return rank, world_size, torch.device(f"cuda:{local_rank}"), mesh


def log(rank, message):
    if rank == 0:
        print(message, flush=True)


# PHITRAIN (3/8) -- = phitrain.models.parallelisms.activation_checkpoint.apply_ac
# Wraps each decoder layer so its activations are recomputed in the backward
# pass instead of being kept. Trades compute for memory. phitrain finds the
# layer list by dotted path (`model.layers`); so do we.
def apply_ac(model):
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i] = checkpoint_wrapper(layers[i], preserve_rng_state=False)


# PHITRAIN (4/8) -- = phitrain.models.parallelisms.data_parallel.apply_fsdp
# fully_shard() is FSDP2. Applied per decoder layer, each rank keeps 1/N of that
# layer's parameters and gathers the rest only while the layer runs.
# reshard_after_forward=False on the last layer keeps it gathered, since the
# backward pass needs it immediately -- the same trick phitrain uses.
def apply_fsdp(model, mesh, param_dtype=torch.bfloat16):
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    layers = model.model.layers
    for i, layer in enumerate(layers):
        fully_shard(
            layer,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=(i < len(layers) - 1),
        )
    fully_shard(model, mesh=mesh, mp_policy=policy)


# PHITRAIN (5/8) -- the original drives its own KVCache and model.reset_kv_cache().
# An HF module exposes past_key_values instead. Same algorithm, different handle:
# this is exactly the "HF defines what the layers are" boundary.
@torch.no_grad()
def sample_response(model, tokenizer, prompt, device, max_new_tokens, temperature, top_p):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    cache = DynamicCache()
    out = model(input_ids.unsqueeze(0), past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits.float(), dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas, num_samples=1)

        token_id = next_token.item()
        generated.append(token_id)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        out = model(next_token, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1]

    full_token_ids = torch.cat(
        [input_ids, torch.tensor(generated, device=device, dtype=input_ids.dtype)]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


# PHITRAIN (6/8) -- the original calls model(...) and gets a tensor back.
# An HF model returns a ModelOutput, so the logits come off .logits.
def sequence_logprob(model, token_ids, prompt_len):
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
    extracted = extract_final_candidate(answer_text, fallback=None)
    if not extracted:
        return 0.0
    return float(grade_answer(extracted, ground_truth))


# UNCHANGED from submit_02_baseline.py. This is the point of the exercise: the
# algorithm is identical whether the model came from a .pth or from HF, and
# whether or not its parameters are sharded across four GPUs.
def compute_grpo_loss(model, tokenizer, example, device, num_rollouts, max_new_tokens,
                      temperature, top_p):
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()
    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model, tokenizer, prompt, device, max_new_tokens, temperature, top_p
        )
        samples.append((token_ids, prompt_len))
        roll_rewards.append(reward_rlvr(text, example["answer"]))
    if was_training:
        model.train()

    for token_ids, prompt_len in samples:
        roll_logps.append(sequence_logprob(model, token_ids, prompt_len))

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)
    loss = -(advantages.detach() * logps).mean()

    return loss, roll_rewards, advantages


def main():
    parser = argparse.ArgumentParser(description="GRPO baseline with an HF model + phitrain-style parallelism.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-5)
    # Rollouts are sampled with multinomial, so without this the run is not
    # reproducible. submit_02_baseline.py has always had --seed; this file
    # needs it for the same reason.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp", action="store_true", help="Shard parameters with FSDP2")
    parser.add_argument("--activation-ckpt", action="store_true", help="Recompute activations")
    args = parser.parse_args()

    # Seed per rank: identical seeds would make every rank sample the same
    # rollouts, which defeats the point of giving each rank its own prompt.
    torch.manual_seed(args.seed)

    rank, world_size, device, mesh = setup_distributed()
    torch.manual_seed(args.seed + rank)
    log(rank, f"world_size={world_size}  device={device}  fsdp={args.fsdp}  ac={args.activation_ckpt}")

    # PHITRAIN (2/8) -- the whole model-loading change. The original was:
    #     model = Qwen3Model(QWEN_CONFIG_06_B)
    #     model.load_state_dict(torch.load("qwen3/qwen3-0.6B-base.pth"))
    # HF carries the architecture in config.json, so no hard-coded config dict.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    log(rank, f"loaded {args.model}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # The passes run in phitrain's order: AC before FSDP, so the checkpoint
    # wrapper is inside the sharded unit rather than around it.
    if args.activation_ckpt:
        apply_ac(model)
    if args.fsdp and mesh is not None:
        apply_fsdp(model, mesh["dp"])
        log(rank, "sharded: params are now DTensors, 1/%d per rank" % world_size)
    else:
        model.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    math_data = load_math_train()

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        t0 = time.perf_counter()

        # PHITRAIN (7/8) -- data parallelism: each rank takes a DIFFERENT example
        # and forms its own GRPO group. FSDP averages the gradients across ranks,
        # which is what `advantage_normalization: prompt` means in the SWE config.
        idx = (step * world_size + rank) % len(math_data)
        loss, rewards, advantages = compute_grpo_loss(
            model, tokenizer, math_data[idx], device,
            args.num_rollouts, args.max_new_tokens, args.temperature, args.top_p,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        reward_avg = torch.tensor(rewards, device=device).mean()
        if world_size > 1:
            dist.all_reduce(reward_avg, op=dist.ReduceOp.AVG)
        log(rank, f"[Step {step+1}/{args.steps}] loss={loss.item():+.4f} "
                  f"reward_avg={reward_avg.item():.3f} {time.perf_counter()-t0:.1f}s")

    # PHITRAIN (8/8) -- the number the flags actually move.
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"[rank {rank}] peak CUDA memory: {peak:.2f} GiB", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
