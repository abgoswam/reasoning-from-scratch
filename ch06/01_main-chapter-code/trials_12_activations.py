# Concept: what autograd "retains" and why grad-enabled generation blows up
#
# Companion to trials_06_logprobs.py. Answers: why does sample_response run
# under @torch.no_grad(), and why is the logprob recomputed in one
# teacher-forced pass instead of reusing the generation-time logits?
#
# Pure toy tensors -- no model, no GPU, runs in a second on CPU.
#
#   Part 1: a saved tensor, seen directly on the graph node
#   Part 2: retained bytes with grad on vs. off
#   Part 3: the KV-cache concat, incremental vs. full-sequence

import torch


def retained_bytes(out):
    """Walk the autograd graph behind `out` and sum bytes of saved tensors."""
    nodes, storages, total = set(), set(), 0
    stack = [out.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in nodes:
            continue
        nodes.add(node)
        for attr in dir(node):
            if not attr.startswith("_saved_"):
                continue
            try:
                value = getattr(node, attr)
            except Exception:
                continue
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                if storage.data_ptr() not in storages:
                    storages.add(storage.data_ptr())
                    total += storage.nbytes()
        for nxt, _ in node.next_functions:
            stack.append(nxt)
    return total


def mib(n):
    return f"{n / 1024**2:8.2f} MiB"


# ---------------------------------------------------------------------------
# Part 1 -- an "activation" is a forward tensor the backward formula needs.
#
#   Y = Q @ K.T   ->   dL/dQ = dL/dY @ K   and   dL/dK = (dL/dY).T @ Q
#
# Both rules mention Q and K, so autograd stashes them on the graph node.
# They stay in memory from the forward pass until backward consumes them.
# ---------------------------------------------------------------------------
def part1_what_gets_saved():
    print("=" * 70)
    print("PART 1  what autograd saves")
    print("=" * 70)

    q = torch.randn(4, 8, requires_grad=True)
    k = torch.randn(4, 8, requires_grad=True)
    y = q @ k.T

    saved = [a for a in dir(y.grad_fn) if a.startswith("_saved_")
             and torch.is_tensor(getattr(y.grad_fn, a, None))]
    print(f"y.grad_fn                : {y.grad_fn}")
    print(f"  saved tensors          : {saved}")
    print(f"  _saved_self  is q      : {y.grad_fn._saved_self is q}")
    print(f"  _saved_mat2 shares k   : "
          f"{y.grad_fn._saved_mat2.data_ptr() == k.data_ptr()}")
    print("  -> the matmul node holds BOTH inputs (mat2 is the k.T view,")
    print("     same underlying storage), so neither can be freed\n")

    x = torch.randn(4, 8, requires_grad=True)
    relu_out = torch.relu(x)
    softmax_out = torch.softmax(x, dim=-1)
    print(f"relu saves its OUTPUT    : "
          f"{torch.equal(relu_out.grad_fn._saved_result, relu_out)}")
    print(f"softmax saves its OUTPUT : "
          f"{torch.equal(softmax_out.grad_fn._saved_result, softmax_out)}")
    print("  -> some ops save the input, some the output; either way it stays\n")

    with torch.no_grad():
        y_nograd = q @ k.T
    print(f"under no_grad, grad_fn   : {y_nograd.grad_fn}")
    print("  -> no node, no saved tensors, memory freed immediately\n")


# ---------------------------------------------------------------------------
# Part 2 -- the same computation, with and without grad. Only the bookkeeping
# differs: identical FLOPs, wildly different peak memory.
# ---------------------------------------------------------------------------
def part2_grad_on_vs_off():
    print("=" * 70)
    print("PART 2  retained bytes: grad on vs. off")
    print("=" * 70)

    layers = [torch.nn.Linear(1024, 1024) for _ in range(8)]
    x = torch.randn(64, 1024)

    out = x
    for layer in layers:
        out = torch.relu(layer(out))
    print(f"grad ON   retained: {mib(retained_bytes(out))}  "
          f"(8 layers x [64,1024] inputs + relu outputs)")

    with torch.no_grad():
        out_nograd = x
        for layer in layers:
            out_nograd = torch.relu(layer(out_nograd))
    print(f"grad OFF  retained: {mib(retained_bytes(out_nograd))}  "
          f"(nothing -- out.grad_fn is None)")
    print("  -> this is exactly what @torch.no_grad() buys sample_response\n")


# ---------------------------------------------------------------------------
# Part 3 -- the real reason generation cannot keep its graph.
#
# qwen3.py:203 rebuilds the whole key tensor every decode step:
#     keys = torch.cat([prev_k, keys_new], dim=2)
# and qwen3.py:215 feeds it to a matmul, which saves it. So step t retains a
# length-t tensor: 1 + 2 + ... + T = O(T^2) instead of O(T).
# ---------------------------------------------------------------------------
def toy_attention(q_t, keys):
    scores = q_t @ keys.transpose(-2, -1)
    return torch.softmax(scores, dim=-1) @ keys


def part3_kv_cache_blowup(steps=64, n_heads=16, head_dim=128):
    print("=" * 70)
    print(f"PART 3  KV cache: incremental vs. full-sequence ({steps} tokens)")
    print("=" * 70)

    torch.manual_seed(0)
    tokens = torch.randn(steps, n_heads, 1, head_dim, requires_grad=True)

    cache, per_step = None, []
    for t in range(steps):
        k_new = tokens[t]
        cache = k_new if cache is None else torch.cat([cache, k_new], dim=-2)
        per_step.append(toy_attention(tokens[t], cache))

    # Every step's output must stay reachable -- generation needs a logprob at
    # each position, so no step's graph can be dropped.
    out = torch.stack(per_step).sum()
    incremental = retained_bytes(out)
    print(f"incremental (grad ON) : {mib(incremental)}   "
          f"step t saves a length-t tensor")

    full_keys = tokens.squeeze(2).transpose(0, 1)
    full_out = toy_attention(full_keys, full_keys).sum()
    full = retained_bytes(full_out)
    print(f"full-sequence (grad ON): {mib(full)}   saved once")
    print(f"ratio                  : {incremental / full:6.1f}x at "
          f"{steps} tokens\n")

    print("same accounting for Qwen3-0.6B (28 layers, 16 heads, d=128, bf16),")
    print("counting saved K, V, queries and attention weights per layer:")
    print(f"  {'tokens':>7} {'incremental':>14} {'full-seq':>13} {'ratio':>8}")
    for length in (64, 128, 256, 512):
        pairs = length * (length + 1) // 2
        inc = (2 * 16 * 128 * 2 * pairs + 16 * 2 * pairs
               + 16 * 128 * 2 * length) * 28
        seq = (2 * 16 * 128 * 2 * length + 16 * 2 * length * length
               + 16 * 128 * 2 * length) * 28
        print(f"  {length:>7} {inc / 1024**3:>11.2f} GiB "
              f"{seq / 1024**3:>9.3f} GiB {inc / seq:>7.0f}x")

    print("\n  Both sides are quadratic -- the full pass has one [H,T,T]")
    print("  attention matrix per layer -- so the ratio grows sub-linearly.")
    print("  But the constant is brutal: at 512 tokens the incremental graph")
    print("  needs ~28 GiB before MLP activations or the [T,151936] logits.")
    print("  Hence: generate under no_grad, then recompute the logprob in")
    print("  ONE teacher-forced pass (trials_00.py sequence_logprob).")


def main(steps=64):
    part1_what_gets_saved()
    part2_grad_on_vs_off()
    part3_kv_cache_blowup(steps=steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64,
                        help="decode steps to simulate in Part 3")
    main(steps=parser.parse_args().steps)
